import os
import configparser
from datetime import datetime
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
import numpy as np

# Environment & utilities (HPC environment)
from HPCSimPickJobs import *  # provides MAX_QUEUE_SIZE, JOB_FEATURES, run_win, green_win, action2_num, delayMaxJobNum, HPCEnv

# Simplified helper: convert flat observation -> torch tensor [1, obs_len]
def reconstruct_state_tensor(o, device=None):
    """
    Accepts a flat observation (list / numpy array / torch tensor) with layout:
      [ job_block (MAX_QUEUE_SIZE * JOB_FEATURES),
        running_block (run_win * 2),
        green_block  (green_win + 8) ]
    Returns: torch.FloatTensor shaped [1, obs_len]
    """
    arr = np.array(o, dtype=np.float32).reshape(-1)
    expected = MAX_QUEUE_SIZE * JOB_FEATURES + run_win * 2 + (green_win + 8)
    if arr.size != expected:
        raise AssertionError(f"Unexpected observation length {arr.size}, expected {expected}")
    tensor = torch.from_numpy(arr).float().unsqueeze(0)
    return tensor.to(device) if device is not None else tensor

# Per-step reward calculator simplified to only two methods:
#  - training reward: co2_wait_combined
#  - carbon evaluation: co2_direct
class PerStepRewardCalculator:
    def __init__(self, config):
        self.config = config
        # Force defaults as requested, but still read config to get weights/windows
        self.reward_function = 'co2_wait_combined'
        self.carbon_reward_function = 'co2_direct'
        self.eta = float(config.get('GAS-MARL setting', 'eta', fallback=0.5))
        self.carbon_weight = float(config.get('reward parameters', 'carbon_weight', fallback=1.0))
        self.wait_weight = float(config.get('reward parameters', 'wait_weight', fallback=1.0))
        self.max_carbon_intensity = float(config.get('algorithm constants', 'max_carbon_intensity', fallback=500.0))
        self.max_wait_time = float(config.get('algorithm constants', 'max_wait_time', fallback=43200.0))
        # Carbon intensity helper
        try:
            from greenPower import carbon_intensity
            carbon_year = int(config.get('general setting', 'carbon_year', fallback=2021))
            green_win_cfg = int(config.get('GAS-MARL setting', 'green_win', fallback=green_win))
            self.carbon_intensity = carbon_intensity(green_win_cfg, carbon_year)
        except Exception:
            self.carbon_intensity = None

    def set_carbon_offset(self, offset):
        if self.carbon_intensity:
            self.carbon_intensity.setStartOffset(offset)

    def calculate_step_reward(self, job, env_context=None):
        """
        Returns dict with 'carbon_reward', 'wait_reward', 'total_reward'.
        Implemented using only co2_wait_combined (training). co2_direct used for carbon-only computations.
        """
        if not hasattr(job, 'scheduled_time') or job.scheduled_time == -1:
            return {'carbon_reward': 0.0, 'wait_reward': 0.0, 'total_reward': 0.0}

        # Carbon penalty for the scheduled job (co2_direct)
        carbon_penalty = self._calculate_co2_direct_reward(job)

        # Wait penalties: for all queued unscheduled jobs, measured at current_time
        total_wait_penalty = 0.0
        current_time = None
        env = None
        if env_context and 'env' in env_context:
            env = env_context['env']
            current_time = env_context.get('current_time', env.current_timestamp)

            if hasattr(env, 'job_queue'):
                for qj in env.job_queue:
                    if not hasattr(qj, 'scheduled_time') or qj.scheduled_time == -1:
                        wait_time = max(0.0, current_time - qj.submit_time)
                        bounded_slowdown = max(1.0, float(wait_time + qj.run_time) / max(qj.run_time, 10))
                        total_wait_penalty += bounded_slowdown
        else:
            # fallback: only scheduled job wait
            current_time = job.scheduled_time
            wait_time = max(0.0, job.scheduled_time - job.submit_time)
            bounded_slowdown = max(1.0, float(wait_time + job.run_time) / max(job.run_time, 10))
            total_wait_penalty = bounded_slowdown

        wait_component = - total_wait_penalty * self.eta * self.wait_weight
        total = wait_component + carbon_penalty  # carbon_penalty already negative
        return {'carbon_reward': carbon_penalty, 'wait_reward': wait_component, 'total_reward': total}

    def _calculate_co2_direct_reward(self, job):
        """
        Compute actual CO2 emissions for the scheduled job and return negative penalty.
        """
        if not self.carbon_intensity:
            return 0.0
        start_time = job.scheduled_time
        end_time = start_time + job.run_time
        power = getattr(job, 'power', 0)
        carbon_consideration = getattr(job, 'carbon_consideration', 0)
        co2_emissions = self.carbon_intensity.getCarbonEmissions(power, start_time, end_time)
        weighted = co2_emissions * carbon_consideration
        # scale down to keep magnitudes reasonable
        return -(weighted / 100000.0) * self.carbon_weight

# Simple rollout buffer (keeps same interface names used later)
class Buffer():
    def __init__(self):
        self.states = []
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.Returns = []
        self.advantages = []
        self.job_inputs = []

    def clear_buffer(self):
        self.__init__()

    def store_buffer(self, state, mask1, mask2, action1, action2, log_prob1, log_prob2, Return, advantage, job_input, nums):
        self.states.extend(state)
        self.masks1.extend(mask1)
        self.masks2.extend(mask2)
        self.actions1.extend(action1)
        self.actions2.extend(action2)
        self.log_probs1.extend(log_prob1)
        self.log_probs2.extend(log_prob2)
        self.Returns.extend(Return)
        self.advantages.extend(advantage)
        self.job_inputs.extend(job_input)

# Simple MLP actor & critic that accept flat observation
class ActorNet(nn.Module):
    def __init__(self, obs_dim, d_model=128, hidden_dim=128):
        super(ActorNet, self).__init__()
        self.d_model = d_model
        # Encoders for each observation block
        self.JobEncoder = nn.Sequential(
            nn.Linear(MAX_QUEUE_SIZE * JOB_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
        )
        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(run_win * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
        )
        self.GreenEncoder = nn.Sequential(
            nn.Linear(green_win + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
        )
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
        )
        # Heads
        # Add one extra output for NO-OP (wait) action
        self.head_job = nn.Linear(hidden_dim, MAX_QUEUE_SIZE + 1)
        self.head_delay = nn.Linear(hidden_dim, action2_num)

    def forward(self, x):
        B = x.size(0)
        ptr = 0
        job_block_len = MAX_QUEUE_SIZE * JOB_FEATURES
        running_block_len = run_win * 2
        green_block_len = green_win + 8

        job_block = x[:, ptr:ptr + job_block_len]
        ptr += job_block_len
        running_block = x[:, ptr:ptr + running_block_len]
        ptr += running_block_len
        green_block = x[:, ptr:ptr + green_block_len]

        job_emb = self.JobEncoder(job_block)
        running_emb = self.RunningJobEncoder(running_block)
        green_emb = self.GreenEncoder(green_block)

        fused = self.fusion(torch.cat([job_emb, running_emb, green_emb], dim=-1))
        logits1 = self.head_job(fused)
        logits2 = self.head_delay(fused)
        return logits1, logits2

class CriticNet(nn.Module):
    def __init__(self, obs_dim, d_model=128, hidden_dim=128):
        super(CriticNet, self).__init__()
        self.d_model = d_model
        self.JobEncoder = nn.Sequential(
            nn.Linear(MAX_QUEUE_SIZE * JOB_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
        )
        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(run_win * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
        )
        self.GreenEncoder = nn.Sequential(
            nn.Linear(green_win + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        B = x.size(0)
        ptr = 0
        job_block_len = MAX_QUEUE_SIZE * JOB_FEATURES
        running_block_len = run_win * 2
        green_block_len = green_win + 8

        job_block = x[:, ptr:ptr + job_block_len]
        ptr += job_block_len
        running_block = x[:, ptr:ptr + running_block_len]
        ptr += running_block_len
        green_block = x[:, ptr:ptr + green_block_len]

        job_emb = self.JobEncoder(job_block)
        running_emb = self.RunningJobEncoder(running_block)
        green_emb = self.GreenEncoder(green_block)

        fused = torch.cat([job_emb, running_emb, green_emb], dim=-1)
        return self.fusion(fused)

class PPO():
    def __init__(self, batch_size=10, obs_dim=None, device='cpu', debug=False, config=None):
        assert obs_dim is not None
        self.obs_dim = obs_dim
        # Accept either a torch.device or a string (e.g., 'cpu'/'cuda')
        if isinstance(device, torch.device):
            self.device = device
        else:
            try:
                self.device = torch.device(device)
            except Exception:
                # fallback to CPU
                self.device = torch.device('cpu')
        self.debug = debug
        self.d_model = int(config.get('model', 'd_model', fallback=128)) if config else 128
        self.actor_net = ActorNet(obs_dim, d_model=self.d_model, hidden_dim=128).to(device)
        self.critic_net = CriticNet(obs_dim, d_model=self.d_model, hidden_dim=128).to(device)
        self.gamma = 1.0
        self.lam = 0.97
        self.ppo_update_time = 8
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.entropy_coefficient = 0.0
        self.minibatch_fraction = 1.0
        if config:
            if config.has_option('ppo', 'gamma'):
                self.gamma = float(config.get('ppo', 'gamma'))
            if config.has_option('ppo', 'lam'):
                self.lam = float(config.get('ppo', 'lam'))
            if config.has_option('ppo', 'ppo_update_time'):
                self.ppo_update_time = int(config.get('ppo', 'ppo_update_time'))
            if config.has_option('ppo', 'clip_param'):
                self.clip_param = float(config.get('ppo', 'clip_param'))
            if config.has_option('ppo', 'max_grad_norm'):
                self.max_grad_norm = float(config.get('ppo', 'max_grad_norm'))
            if config.has_option('ppo', 'entropy_coef'):
                self.entropy_coefficient = float(config.get('ppo', 'entropy_coef'))
            if config.has_option('ppo', 'minibatch_fraction'):
                self.minibatch_fraction = float(config.get('ppo', 'minibatch_fraction'))
        # Optimizers
        actor_lr = float(config.get('ppo', 'actor_lr', fallback=0.0001)) if config else 0.0001
        critic_lr = float(config.get('ppo', 'critic_lr', fallback=0.0005)) if config else 0.0005
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr, eps=1e-6)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr, eps=1e-6)
        self.batch_size = batch_size
        self.buffer = Buffer()
        self.states = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.rewards_seq = []
        self.actions1 = []
        self.actions2 = []
        self.values = []
        self.masks1 = []
        self.masks2 = []
        self.job_inputs = []
        self.step_rewards = []
        self.step_reward_components = []
        self.per_step_reward_calculator = None
        if config:
            self.per_step_reward_calculator = PerStepRewardCalculator(config)
        # Reward scaling factor used only for training computations (advantages/returns).
        # Keep reporting in original units but scale rewards before discounting to improve numeric stability.
        # Increase default reward_scale to 1000 for better numerical stability when raw rewards are small.
        self.reward_scale = float(config.get('ppo', 'reward_scale', fallback=1000.0)) if config else 1000.0

    def choose_action(self, state, mask1, mask2):
        # state: [1, obs_dim], masks: torch.FloatTensor [1, MAX_QUEUE_SIZE+1], [1, action2_num]
        # This function returns tensors shaped for single-step storage (each tensor has batch dim = 1)
        with torch.no_grad():
            # Actor produces logits with shape [1, action_dim]; masks already have batch dim
            logits1, logits2 = self.actor_net(state)  # [1, MAX_QUEUE_SIZE+1], [1, action2_num]
            logits1 = logits1 - mask1 * 1e9
            logits2 = logits2 - mask2 * 1e9
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            dist1 = Categorical(probs=probs1)
            dist2 = Categorical(probs=probs2)
            ac1 = dist1.sample()   # shape [1]
            ac2 = dist2.sample()   # shape [1]
            log_prob1 = dist1.log_prob(ac1)  # shape [1]
            log_prob2 = dist2.log_prob(ac2)  # shape [1]
            value = self.critic_net(state).squeeze(-1)  # shape [1]
            # Extract job_input by slicing flat state
            job_block_len = MAX_QUEUE_SIZE * JOB_FEATURES
            job_idx = int(ac1.item())
            # If agent selects the NO-OP slot (index == MAX_QUEUE_SIZE), return zero job_input
            if job_idx >= MAX_QUEUE_SIZE:
                job_input = torch.zeros(1, JOB_FEATURES, device=state.device)
            else:
                start = job_idx * JOB_FEATURES
                end = start + JOB_FEATURES
                job_input = state[:, start:end]  # shape [1, JOB_FEATURES]
            # Return shapes: ac1[1], log_prob1[1], ac2[1], log_prob2[1], value[1], job_input[1, JOB_FEATURES]
            return ac1, log_prob1, ac2, log_prob2, value, job_input

    def act_job(self, states, masks1, actions1):
        # Batch-compatible computation of log-prob and entropy for action1.
        # states: [B, obs_dim], masks1: [B, MAX_QUEUE_SIZE+1] (or [B, MAX_QUEUE_SIZE+1,1]), actions1: [B] or [B,1]
        logits1, _ = self.actor_net(states)  # [B, MAX_QUEUE_SIZE+1]
        # normalize mask shape
        mask1_proc = masks1.squeeze(-1) if masks1.dim() == 3 else masks1
        logits1 = logits1 - mask1_proc * 1e9
        probs1 = F.softmax(logits1, dim=-1)
        dist1 = Categorical(probs=probs1)
        actions1_proc = actions1.squeeze(-1) if actions1.dim() > 1 else actions1
        log_prob1 = dist1.log_prob(actions1_proc)  # [B]
        entropy1 = dist1.entropy()  # [B]
        return log_prob1, entropy1

    def act_exc(self, states, masks2, job_input, actions2):
        # Batch-compatible computation of log-prob and entropy for action2.
        # states: [B, obs_dim], masks2: [B, action2_num], actions2: [B] or [B,1]
        _, logits2 = self.actor_net(states)  # [B, action2_num]
        mask2_proc = masks2.squeeze(-1) if masks2.dim() == 3 else masks2
        logits2 = logits2 - mask2_proc * 1e9
        probs2 = F.softmax(logits2, dim=-1)
        dist2 = Categorical(probs=probs2)
        actions2_proc = actions2.squeeze(-1) if actions2.dim() > 1 else actions2
        log_prob2 = dist2.log_prob(actions2_proc)  # [B]
        entropy2 = dist2.entropy()  # [B]
        return log_prob2, entropy2

    def normalize(self, advantages):
        # Standardize advantages and clamp to avoid extreme values that destabilize training.
        adv = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-9)
        return torch.clamp(adv, -10.0, 10.0)

    def remember(self, state, value, log_prob1, log_prob2, action1, action2, reward, mask1, mask2, device, job_input):
        self.rewards_seq.append(reward)
        self.states.append(state.to("cpu"))
        self.log_probs1.append(log_prob1.to("cpu"))
        self.log_probs2.append(log_prob2.to("cpu"))
        self.values.append(value.to("cpu"))
        self.actions1.append(action1.to("cpu"))
        self.actions2.append(action2.to("cpu"))
        self.masks1.append(mask1.to("cpu"))
        self.masks2.append(mask2.to("cpu"))
        self.job_inputs.append(job_input.to("cpu"))

    def clear_memory(self):
        self.rewards_seq = []
        self.states = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.values = []
        self.actions1 = []
        self.actions2 = []
        self.masks1 = []
        self.masks2 = []
        self.job_inputs = []
        self.step_rewards = []
        self.step_reward_components = []

    def calculate_per_step_reward(self, job, env_context=None):
        if self.per_step_reward_calculator:
            comp = self.per_step_reward_calculator.calculate_step_reward(job, env_context)
            self.step_reward_components.append(comp)
            self.step_rewards.append(comp['total_reward'])
            # Per-step prints removed to reduce console noise (was printed when self.debug was True)
            return comp
        # fallback
        wait_time = job.scheduled_time - job.submit_time
        bounded_slowdown = max(1.0, float(wait_time + job.run_time) / max(job.run_time, 10))
        wait_penalty = -bounded_slowdown * 0.1
        return {'carbon_reward': 0.0, 'wait_reward': wait_penalty, 'total_reward': wait_penalty}

    def calculate_delay_reward(self, env_context):
        # delay reward: wait penalty for queued jobs at future_time
        if not env_context or 'env' not in env_context:
            return {'carbon_reward': 0.0, 'wait_reward': 0.0, 'total_reward': 0.0}
        env = env_context['env']
        current_time = env_context.get('current_time', env.current_timestamp)
        action2 = env_context.get('action2', 0)
        future_time = current_time
        if action2 > 0:
            # fixed delay list from config
            try:
                config = configparser.ConfigParser()
                config.read('./configFile/config.ini')
                delaytimelist = eval(config.get('GAS-MARL setting', 'delaytimelist', fallback='[1100,2200,5400,10800,21600,43200,86400]'))
            except Exception:
                delaytimelist = [1100,2200,5400,10800,21600,43200,86400]
            delay_max_job_num = getattr(env, 'delayMaxJobNum', 5)
            if action2 <= delay_max_job_num:
                if hasattr(env, 'running_jobs') and len(env.running_jobs) >= action2:
                    running_job = env.running_jobs[action2 - 1]
                    estimated_completion = running_job.scheduled_time + running_job.run_time
                    future_time = max(current_time, estimated_completion)
            else:
                delay_index = action2 - delay_max_job_num - 1
                if 0 <= delay_index < len(delaytimelist):
                    future_time = current_time + delaytimelist[delay_index]
        total_wait_penalty = 0.0
        if hasattr(env, 'job_queue'):
            for qj in env.job_queue:
                if not hasattr(qj, 'scheduled_time') or qj.scheduled_time == -1:
                    wait_time = future_time - qj.submit_time
                    if wait_time > 0:
                        bounded_slowdown = max(1.0, float(wait_time + qj.run_time) / max(qj.run_time, 10))
                        total_wait_penalty += bounded_slowdown
        wait_weight = self.per_step_reward_calculator.wait_weight if self.per_step_reward_calculator else 0.1
        wait_reward = - total_wait_penalty * (self.per_step_reward_calculator.eta if self.per_step_reward_calculator else 0.5) * wait_weight
        return {'carbon_reward': 0.0, 'wait_reward': wait_reward, 'total_reward': wait_reward}

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        """
        Compute GAE advantages and discounted returns.
        Rewards are scaled by self.reward_scale for numerical stability during training,
        but external reporting (ep_ret) remains in original units.
        """
        # Scale rewards for training stability
        scaled_rews = np.array(self.rewards_seq, dtype=float) * float(getattr(self, 'reward_scale', 1.0))
        scaled_last = float(last_val) * float(getattr(self, 'reward_scale', 1.0))
        rews = np.append(scaled_rews, scaled_last)
        # Values from critic (cpu numpy)
        values = torch.cat(self.values, dim=0)
        values = values.squeeze(dim=-1)
        vals = np.append(np.array(values.cpu(), dtype=float), scaled_last)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = self.discount_cumsum(deltas, self.gamma * self.lam)
        ret = self.discount_cumsum(rews, self.gamma)[:-1]
        return adv, ret

    def storeIntoBuffter(self, reward):
        advantages, returns = self.finish_path(reward)
        returns = returns.tolist()
        advantages = advantages.tolist()
        self.buffer.store_buffer(self.states, self.masks1, self.masks2, self.actions1, self.actions2, self.log_probs1, self.log_probs2, returns, advantages, self.job_inputs, len(self.states))

    def compute_value_loss(self, states, returns):
        state_values = self.critic_net(states).squeeze(dim=1)
        value_loss = F.mse_loss(state_values, returns)
        return value_loss

    def compute_actor_loss(self,
                           states,
                           masks1,
                           masks2,
                           actions1,
                           actions2,
                           advantages,
                           old_log_probs1,
                           old_log_probs2,
                           job_input
                           ):
        # Compute new log-probs for the sampled minibatch
        log_probs1, entropy1 = self.act_job(states, masks1, actions1)
        log_probs2, entropy2 = self.act_exc(states, masks2, job_input, actions2)

        # Defensive shape checks to avoid silent broadcasting errors
        # expected shapes: log_probsX: [B], old_log_probsX: [B], advantages: [B]
        if log_probs1.dim() != 1 or log_probs2.dim() != 1:
            raise RuntimeError(f"act_job/act_exc returned unexpected shapes: log_probs1 {log_probs1.shape}, log_probs2 {log_probs2.shape}")
        if old_log_probs1.dim() != 1 or old_log_probs2.dim() != 1:
            raise RuntimeError(f"old_log_probs have unexpected shapes: {old_log_probs1.shape}, {old_log_probs2.shape}")
        if advantages.dim() != 1:
            # allow [B,1] by squeezing
            advantages = advantages.squeeze(-1)
        if old_log_probs1.shape != log_probs1.shape or old_log_probs2.shape != log_probs2.shape or advantages.shape != log_probs1.shape:
            raise RuntimeError(f"Shape mismatch: log_probs1 {log_probs1.shape}, log_probs2 {log_probs2.shape}, old_log_probs1 {old_log_probs1.shape}, old_log_probs2 {old_log_probs2.shape}, advantages {advantages.shape}")

        log_old = old_log_probs1 + old_log_probs2
        log_new = log_probs1 + log_probs2
        logratio = log_new - log_old
        ratio = torch.exp(logratio)
        surr1 = ratio * advantages
        clip_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
        surr2 = clip_ratio * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        entropy = (entropy1 + entropy2) / 2
        entropy_loss = torch.mean(entropy)
        total_loss = policy_loss - self.entropy_coefficient * entropy_loss
        kl_mean = torch.mean(log_old - log_new).detach()
        return total_loss, policy_loss, entropy_loss, kl_mean

    def train(self):
        if not self.buffer.states:
            return {}
        states = torch.cat(self.buffer.states, dim=0).to(self.device)
        masks1 = torch.cat(self.buffer.masks1, dim=0).to(self.device)
        masks2 = torch.cat(self.buffer.masks2, dim=0).to(self.device)
        actions1 = torch.cat(self.buffer.actions1, dim=0).to(self.device)
        log_probs1 = torch.cat(self.buffer.log_probs1, dim=0).to(self.device)
        actions2 = torch.cat(self.buffer.actions2, dim=0).to(self.device)
        log_probs2 = torch.cat(self.buffer.log_probs2, dim=0).to(self.device)
        job_inputs = torch.cat(self.buffer.job_inputs, dim=0).to(self.device)
        returns = torch.tensor(self.buffer.Returns).to(self.device)
        advantages = torch.tensor(self.buffer.advantages).to(self.device)
        advantages_raw = advantages.clone()
        advantages = self.normalize(advantages)
        metrics = {'policy_loss_clipped': [], 'policy_loss_unclipped': [], 'entropy': [], 'approx_kl': [], 'clip_fraction': [], 'value_loss': [], 'explained_variance': []}
        for i in range(self.ppo_update_time):
            effective_batch_size = max(1, int(self.batch_size * self.minibatch_fraction))
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.states))), effective_batch_size, False):
                index_tensor = torch.tensor(index)
                sampled_states = torch.index_select(states, dim=0, index=index_tensor).to(self.device)
                sampled_masks1 = torch.index_select(masks1, dim=0, index=index_tensor).to(self.device)
                sampled_masks2 = torch.index_select(masks2, dim=0, index=index_tensor).to(self.device)
                sampled_actions1 = torch.index_select(actions1, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs1 = torch.index_select(log_probs1, dim=0, index=index_tensor).to(self.device)
                sampled_actions2 = torch.index_select(actions2, dim=0, index=index_tensor).to(self.device)
                sampled_log_probs2 = torch.index_select(log_probs2, dim=0, index=index_tensor).to(self.device)
                sampled_returns = torch.index_select(returns, dim=0, index=index_tensor).to(self.device)
                sampled_advantages = torch.index_select(advantages, dim=0, index=index_tensor).to(self.device)
                sampled_job_inputs = torch.index_select(job_inputs, dim=0, index=index_tensor).to(self.device)

                self.actor_optimizer.zero_grad()
                action_loss, policy_loss, entropy_loss, kl_mean = self.compute_actor_loss(sampled_states, sampled_masks1, sampled_masks2, sampled_actions1, sampled_actions2, sampled_advantages, sampled_log_probs1, sampled_log_probs2, sampled_job_inputs)
                with torch.no_grad():
                    log_probs1_new, _ = self.act_job(sampled_states, sampled_masks1, sampled_actions1)
                    log_probs2_new, _ = self.act_exc(sampled_states, sampled_masks2, sampled_job_inputs, sampled_actions2)
                    log_old = sampled_log_probs1 + sampled_log_probs2
                    log_new = log_probs1_new + log_probs2_new
                    ratio = torch.exp(log_new - log_old)
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    clip_fraction = torch.mean((ratio != clipped_ratio).float()).item()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_net_optimizer.zero_grad()
                value_loss = self.compute_value_loss(sampled_states, sampled_returns)
                with torch.no_grad():
                    state_values = self.critic_net(sampled_states).squeeze(dim=1)
                    y_pred = state_values
                    y_true = sampled_returns
                    var_y = torch.var(y_true)
                    explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
                    explained_variance = explained_var.item()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                metrics['policy_loss_clipped'].append(action_loss.item())
                metrics['policy_loss_unclipped'].append(policy_loss.item())
                metrics['entropy'].append(entropy_loss.item())
                metrics['approx_kl'].append(kl_mean.item())
                metrics['clip_fraction'].append(clip_fraction)
                metrics['value_loss'].append(value_loss.item())
                metrics['explained_variance'].append(explained_variance)

        aggregated = {}
        for k, v in metrics.items():
            aggregated[k] = sum(v) / len(v) if v else 0
        aggregated['adv_mean'] = advantages_raw.mean().item() if len(advantages_raw) > 0 else 0
        aggregated['adv_std'] = advantages_raw.std().item() if len(advantages_raw) > 0 else 0
        return aggregated

    def save_using_model_name(self, model_name_path):
        if not os.path.exists(model_name_path):
            os.makedirs(model_name_path, exist_ok=True)
        torch.save(self.actor_net.state_dict(), os.path.join(model_name_path, "_actor.pkl"))
        torch.save(self.critic_net.state_dict(), os.path.join(model_name_path, "_critic.pkl"))

    def load_using_model_name(self, model_name_path):
        self.actor_net.load_state_dict(torch.load(os.path.join(model_name_path, "_actor.pkl"), map_location=self.device))
        self.critic_net.load_state_dict(torch.load(os.path.join(model_name_path, "_critic.pkl"), map_location=self.device))

    def eval_action(self, o, mask1, mask2):
        # o is flat observation (list/np), mask1 list, mask2 list
        state = reconstruct_state_tensor(o, device=self.device)
        mask1 = np.array(mask1).reshape(1, MAX_QUEUE_SIZE + 1)
        mask1 = torch.FloatTensor(mask1).to(self.device)
        mask2 = np.array(mask2).reshape(1, action2_num)
        mask2 = torch.FloatTensor(mask2).to(self.device)
        with torch.no_grad():
            logits1, logits2 = self.actor_net(state)
            logits1 = logits1.squeeze(0) - mask1.squeeze(0) * 1e9
            logits2 = logits2.squeeze(0) - mask2.squeeze(0) * 1e9
            probs1 = F.softmax(logits1, dim=-1)
            probs2 = F.softmax(logits2, dim=-1)
            dist1 = Categorical(probs=probs1)
            dist2 = Categorical(probs=probs2)
            ac1 = dist1.sample()
            ac2 = dist2.sample()
        return ac1, ac2

# Training orchestration (simplified)
def setup_experiment_directory(workload, experiment_name, description):
    experiment_dir = f"{workload}/MARL_{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{experiment_dir}/final", exist_ok=True)
    if description:
        with open(os.path.join(experiment_dir, "description.txt"), "w") as f:
            f.write(f"Experiment: MARL_{experiment_name}\nWorkload: {workload}\nTimestamp: {datetime.now().isoformat()}\nDescription: {description}\n")
    # snapshot config
    config = configparser.ConfigParser()
    config.read('configFile/config.ini')
    with open(os.path.join(experiment_dir, "config_snapshot.ini"), "w") as f:
        f.write(f"# Configuration snapshot for experiment MARL_{experiment_name}\n# Generated: {datetime.now().isoformat()}\n\n")
        for section in config.sections():
            f.write(f"[{section}]\n")
            for k, v in config.items(section):
                f.write(f"{k} = {v}\n")
            f.write("\n")
    return experiment_dir

def train(workload, backfill, debug=False, experiment_name="", description="", no_score=True, validate_every_n_epochs=False):
    print("Training (simplified) called")
    config = configparser.ConfigParser()
    optuna_config_path = os.environ.get('OPTUNA_CONFIG_PATH')
    if optuna_config_path and os.path.exists(optuna_config_path):
        config.read(optuna_config_path)
    else:
        config.read('configFile/config.ini')

    # Enforce requested defaults and simple checks
    if config.get('carbon setting', 'use_dynamic_window', fallback='True').lower() not in ('1', 'true', 'yes'):
        print("Overriding config: use_dynamic_window -> True")
    # force reward function settings in effect
    print("Enforcing reward_function = co2_wait_combined and carbon_reward_function = co2_direct")

    seed = int(config.get('training parameters', 'seed', fallback=0))
    epochs = int(config.get('training parameters', 'epochs', fallback=1))
    traj_num = int(config.get('training parameters', 'traj_num', fallback=1))

    delaytimelist = eval(config.get('GAS-MARL setting', 'delaytimelist', fallback='[300,600,1200,1800,2400,3000,3600]'))

    experiment_dir = setup_experiment_directory(workload, experiment_name, description)
    print(f"Experiment directory: {experiment_dir}")

    env = HPCEnv(backfill=backfill, debug=debug)
    env.seed(seed)
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, "data", f"{workload}.swf")
    if not os.path.exists(workload_file):
        print(f"ERROR: Workload file not found: {workload_file}")
        exit(1)
    env.my_init(workload_file=workload_file)

    # Device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    obs_len = MAX_QUEUE_SIZE * JOB_FEATURES + run_win * 2 + (green_win + 8)

    batch_size = int(config.get('ppo', 'batch_size', fallback=256))

    ppo = PPO(batch_size=batch_size, obs_dim=obs_len, device=device, debug=debug, config=config)

    for epoch in range(epochs):
        if debug:
            print(f"Starting epoch {epoch + 1}/{epochs}")
        # collect traj_num trajectories
        epoch_start = time.time()
        step_counter = 0
        episode_returns = []
        episode_lengths = []
        epoch_trajectory_rewards = []
        t = 0
        while t < traj_num:
            o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = env.reset(), 0, False, 0, 0, 0, 0, 0, 0
            running_num = 0
            trajectory_rewards = []
            while True:
                # Build mask1 (jobs) same as before by inspecting o's job block
                lst = []
                for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                    job_slice = o[i:i + JOB_FEATURES]
                    padding_pattern1 = [0, 1, 1, 1, 1, 0.5, 0]
                    padding_pattern2 = [1] * JOB_FEATURES
                    if (len(job_slice) == len(padding_pattern1) and all(abs(job_slice[j] - padding_pattern1[j]) < 1e-6 for j in range(len(job_slice)))):
                        lst.append(1)
                    elif (len(job_slice) == len(padding_pattern2) and all(abs(job_slice[j] - padding_pattern2[j]) < 1e-6 for j in range(len(job_slice)))):
                        lst.append(1)
                    else:
                        lst.append(0)
                valid_jobs = lst.count(0)
                # Append NO-OP slot mask (always available, so 0)
                lst_with_noop = lst + [0]

                mask2 = np.zeros(action2_num, dtype=int)
                if running_num < delayMaxJobNum:
                    mask2[running_num + 1 : delayMaxJobNum + 1] = 1

                # state tensor
                state = reconstruct_state_tensor(o, device=device)
                mask1T = torch.FloatTensor(np.array(lst_with_noop).reshape(1, MAX_QUEUE_SIZE + 1)).to(device)
                mask2T = torch.FloatTensor(mask2.reshape(1, action2_num)).to(device)

                action1, log_prob1, action2, log_prob2, value, job_input = ppo.choose_action(state, mask1T, mask2T)

                # step environment
                # Support explicit NO-OP / WAIT action (actor head has MAX_QUEUE_SIZE+1 outputs).
                # If agent chose the NO-OP slot, interpret action2 to decide how to wait and do not call env.step.
                if int(action1.item()) >= MAX_QUEUE_SIZE:
                    # Perform wait according to action2 semantics (same as env.skip1/skip2 semantics)
                    if int(action2.item()) > 0 and int(action2.item()) <= delayMaxJobNum:
                        # try skip1 (wait until the a2-th running job completes or fallback)
                        try:
                            env.skip1(int(action2.item()))
                        except Exception:
                            nextGreenChange = ((env.current_timestamp // 3600) + 1) * 3600
                            env.current_timestamp = max(env.current_timestamp, nextGreenChange)
                            env.cluster.PowerStruc.updateCurrentTime(env.current_timestamp)
                    elif int(action2.item()) > delayMaxJobNum:
                        # fixed delay using delayTimeList
                        try:
                            skipTime = delayTimeList[int(action2.item()) - delayMaxJobNum - 1]
                            env.skip2(skipTime)
                        except Exception:
                            nextGreenChange = ((env.current_timestamp // 3600) + 1) * 3600
                            env.current_timestamp = max(env.current_timestamp, nextGreenChange)
                            env.cluster.PowerStruc.updateCurrentTime(env.current_timestamp)
                    else:
                        # default: advance to next green-change boundary
                        nextGreenChange = ((env.current_timestamp // 3600) + 1) * 3600
                        env.current_timestamp = max(env.current_timestamp, nextGreenChange)
                        env.cluster.PowerStruc.updateCurrentTime(env.current_timestamp)

                    # After waiting, get new observation without scheduling a job
                    o = env.build_observation()
                    r = 0
                    d = False
                    r2 = 0
                    sjf_t = 0
                    f1_t = 0
                    running_num = len(env.running_jobs)
                    greenRwd = 0
                else:
                    o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = env.step(action1.item(), action2.item())

                env_context = {'env': env, 'current_time': env.current_timestamp, 'episode_step': ep_len, 'action1': action1.item(), 'action2': action2.item(), 'epoch': epoch + 1, 'trajectory': t + 1}

                # If action1 corresponds to a real job (not NO-OP)
                if action1.item() < len(env.pairs) and env.pairs[action1.item()][0] is not None and action1.item() < MAX_QUEUE_SIZE:
                    job_for_scheduling = env.pairs[action1.item()][0]
                    if hasattr(job_for_scheduling, 'scheduled_time') and job_for_scheduling.scheduled_time != -1:
                        reward_components = ppo.calculate_per_step_reward(job_for_scheduling, env_context)
                    else:
                        reward_components = ppo.calculate_delay_reward(env_context)
                else:
                    # NO-OP or padding/selecting empty slot -> delay reward
                    reward_components = ppo.calculate_delay_reward(env_context)

                step_reward_total = reward_components['total_reward']
                step_reward_wait = reward_components['wait_reward']
                step_reward_carbon = reward_components['carbon_reward']

                ppo.remember(state, value, log_prob1, log_prob2, action1, action2, step_reward_total, mask1T, mask2T, device, job_input)

                trajectory_rewards.append((step_reward_wait, step_reward_carbon, step_reward_total))
                epoch_trajectory_rewards.append((step_reward_wait, step_reward_carbon, step_reward_total))
                step_counter += 1

                ep_len += 1
                ep_ret += step_reward_total

                if d:
                    t += 1
                    # finish path and store into buffer
                    episode_returns.append(ep_ret)
                    episode_lengths.append(ep_len)
                    ppo.storeIntoBuffter(step_reward_total)
                    ppo.clear_memory()
                    break

        # After collecting trajectories, train
        training_metrics = ppo.train()

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(experiment_dir, "checkpoints", f"epoch_{epoch + 1}")
            ppo.save_using_model_name(checkpoint_path)
            if debug:
                print(f"Saved checkpoint to {checkpoint_path}")

        # clear buffer for next epoch
        ppo.buffer.clear_buffer()

        # Epoch statistics and logging
        epoch_time = time.time() - epoch_start
        steps_per_sec = step_counter / epoch_time if epoch_time > 0 else 0.0
        num_episodes = len(episode_returns)

        # Raw (unscaled) episode stats for human-readable reporting
        avg_ep_reward_raw = float(np.mean(episode_returns)) if num_episodes > 0 else 0.0
        med_ep_reward_raw = float(np.median(episode_returns)) if num_episodes > 0 else 0.0
        avg_ep_len = float(np.mean(episode_lengths)) if num_episodes > 0 else 0.0
        avg_step_reward_raw = float(np.mean([x[2] for x in epoch_trajectory_rewards])) if step_counter > 0 else 0.0
        avg_green_reward = float(np.mean([x[1] for x in epoch_trajectory_rewards])) if step_counter > 0 else 0.0
        avg_wait_reward = float(np.mean([x[0] for x in epoch_trajectory_rewards])) if step_counter > 0 else 0.0

        # Scale rewards for numeric stability in training and reporting (ppo.reward_scale defaults to 100)
        reward_scale = float(getattr(ppo, "reward_scale", 1.0))
        avg_ep_reward = avg_ep_reward_raw * reward_scale
        med_ep_reward = med_ep_reward_raw * reward_scale
        avg_step_reward = avg_step_reward_raw * reward_scale

        print("=" * 80)
        print(f"Epoch {epoch + 1}/{epochs} | time: {epoch_time:.2f}s | steps: {step_counter} | {steps_per_sec:.2f} steps/s")
        print(f"  episodes: {num_episodes} | episode_return_mean: {avg_ep_reward:.6f} | episode_return_median: {med_ep_reward:.6f} | avg_ep_len: {avg_ep_len:.2f}")
        print(f"  avg_epoch_reward (per-step): {avg_step_reward:.6f} | avg_green_reward: {avg_green_reward:.6f} | avg_wait_reward: {avg_wait_reward:.6f}")
        if training_metrics:
            print(f"  policy_loss_clipped: {training_metrics.get('policy_loss_clipped', 0):.6f} | value_loss: {training_metrics.get('value_loss', 0):.6f} | entropy: {training_metrics.get('entropy', 0):.6f} | approx_kl: {training_metrics.get('approx_kl', 0):.6f}")
            print(f"  adv_mean: {training_metrics.get('adv_mean', 0):.6f} | adv_std: {training_metrics.get('adv_std', 0):.6f}")
        print("=" * 80)

        # CSV logging
        training_log_path = os.path.join(experiment_dir, "training_log.csv")
        write_header = not os.path.exists(training_log_path)
        try:
            with open(training_log_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if write_header:
                    writer.writerow(["epoch","episode_return_mean","episode_return_median","avg_epoch_reward","avg_green_reward","avg_wait_reward","steps","steps_per_sec","avg_ep_len","policy_loss_clipped","value_loss","entropy","approx_kl","adv_mean","adv_std"])
                writer.writerow([
                    epoch + 1,
                    f"{avg_ep_reward:.6f}",
                    f"{med_ep_reward:.6f}",
                    f"{avg_step_reward:.6f}",
                    f"{avg_green_reward:.6f}",
                    f"{avg_wait_reward:.6f}",
                    step_counter,
                    f"{steps_per_sec:.2f}",
                    f"{avg_ep_len:.2f}",
                    f"{training_metrics.get('policy_loss_clipped', 0):.6f}",
                    f"{training_metrics.get('value_loss', 0):.6f}",
                    f"{training_metrics.get('entropy', 0):.6f}",
                    f"{training_metrics.get('approx_kl', 0):.6f}",
                    f"{training_metrics.get('adv_mean', 0):.6f}",
                    f"{training_metrics.get('adv_std', 0):.6f}"
                ])
        except Exception as e:
            print(f"Warning: failed to write training log CSV: {e}")

    # Save final model
    final_model_path = os.path.join(experiment_dir, "final")
    ppo.save_using_model_name(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    return experiment_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='lublin_256', help='Workload dataset to use')
    parser.add_argument('--backfill', type=int, default=0, help='Backfill strategy (0/1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--name', type=str, required=True, help='Experiment name (e.g., ED12)')
    parser.add_argument('--description', type=str, default='', help='Description of the experiment')
    args = parser.parse_args()
    train(args.workload, args.backfill, args.debug, args.name, args.description)
