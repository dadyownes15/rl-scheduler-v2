import os
import csv
import shutil
import configparser
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
import numpy as np
import pandas as pd

# Import score calculation functionality
try:
    from calculate_score import run_carbon_emissions_regression, run_wait_time_regression, calculate_score
    SCORE_CALCULATION_AVAILABLE = True
except ImportError:
    SCORE_CALCULATION_AVAILABLE = False
    print("Warning: Score calculation not available. calculate_score.py not found.")

# Import validation functionality for score calculation
try:
    from validate import load_marl_model, run_enhanced_marl_validation, EnhancedJobTracker, save_enhanced_validation_results
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Warning: Validation functionality not available for score calculation.")

from HPCSimPickJobs import *

class Buffer():
    def __init__(self):
        self.buffer_num = 0
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
        self.buffer_num = 0
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

    def store_buffer(self, state, mask1, mask2, action1, action2, log_prob1, log_prob2, Return, advantage, job_input,
                     nums):
        self.buffer_num = self.buffer_num + nums
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


class ActorNet(nn.Module):

    def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3):
        super(ActorNet, self).__init__()
        self.d_model = 128

        self.num_inputs1 = num_inputs1
        self.featureNum1 = featureNum1
        self.num_inputs2 = num_inputs2
        self.featureNum2 = featureNum2
        self.num_inputs3 = num_inputs3
        self.featureNum3 = featureNum3

        self.embedding = nn.Linear(in_features=JOB_FEATURES, out_features=self.d_model)
        self.JobEncoder = nn.Sequential(
            nn.Linear(self.featureNum1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.GreenEncoder = nn.Sequential(
            nn.Linear(self.featureNum3, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(self.featureNum2, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.decoder1 = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        self.hidden = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, JOB_FEATURES),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()

        self.decoder2 = nn.Sequential(
            nn.Linear((self.num_inputs1 + self.num_inputs2 + self.num_inputs3 + 1) * JOB_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, action2_num),
        )

    def forward(self, x):
        job = x[:, :self.num_inputs1, :self.featureNum1]
        run = x[:, self.num_inputs1:self.num_inputs1 + self.num_inputs2, :self.featureNum2]
        green = x[:, self.num_inputs1 + self.num_inputs2:self.num_inputs1 + self.num_inputs2 + self.num_inputs3,
                :self.featureNum3]
        job = self.JobEncoder(job)
        run = self.RunningJobEncoder(run)
        green = self.GreenEncoder(green)

        return job, run, green

    def getActionn1(self, x, mask):
        encoder_out, _, _ = self.forward(x)
        logits = self.decoder1(encoder_out)

        logits = logits.squeeze(dim=-1)

        logits = logits - mask * 1e9
        probs = F.softmax(logits, dim=-1)

        return probs

    def getAction2(self, x, mask, job_input):
        job, run, green = self.forward(x)
        job_input = self.embedding(job_input)
        encoder_out = torch.cat([job, run, green, job_input], dim=1)
        encoder_out = self.hidden(encoder_out)
        encoder_out = self.flatten(encoder_out)
        logits = self.decoder2(encoder_out)
        logits = logits - mask * 1e9
        probs = F.softmax(logits, dim=-1)

        return probs


class CriticNet(nn.Module):

    def __init__(self, num_inputs1, featureNum1, num_inputs2, featureNum2, num_inputs3, featureNum3):
        super(CriticNet, self).__init__()
        self.d_model = 128

        self.num_inputs1 = num_inputs1
        self.featureNum1 = featureNum1
        self.num_inputs2 = num_inputs2
        self.featureNum2 = featureNum2
        self.num_inputs3 = num_inputs3
        self.featureNum3 = featureNum3

        self.JobEncoder = nn.Sequential(
            nn.Linear(self.featureNum1, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.GreenEncoder = nn.Sequential(
            nn.Linear(self.featureNum3, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.RunningJobEncoder = nn.Sequential(
            nn.Linear(self.featureNum2, 64),
            nn.ReLU(),
            nn.Linear(64, self.d_model),
            nn.ReLU(),
        )

        self.hidden = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, JOB_FEATURES),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear((self.num_inputs1 + self.num_inputs2 + self.num_inputs3) * JOB_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        job = x[:, :self.num_inputs1, :self.featureNum1]
        run = x[:, self.num_inputs1:self.num_inputs1 + self.num_inputs2, :self.featureNum2]
        green = x[:, self.num_inputs1 + self.num_inputs2:self.num_inputs1 + self.num_inputs2 + self.num_inputs3,
                :self.featureNum3]
        green = self.GreenEncoder(green)
        job = self.JobEncoder(job)
        run = self.RunningJobEncoder(run)

        con = torch.cat([job, run, green], dim=1)

        con = self.hidden(con)
        con = self.flatten(con)
        value = self.out(con)
        return value


class PPO():
    def __init__(self, batch_size=10, inputNum_size=[], featureNum_size=[],
                 device='cpu', debug=False):
        super(PPO, self).__init__()
        self.inputNum_size = inputNum_size
        self.featureNum_size = featureNum_size
        self.actor_net = ActorNet(inputNum_size[0], featureNum_size[0], inputNum_size[1], featureNum_size[1],
                                  inputNum_size[2], featureNum_size[2])
        self.critic_net = CriticNet(inputNum_size[0], featureNum_size[0], inputNum_size[1], featureNum_size[1],
                                    inputNum_size[2], featureNum_size[2])
        self.device = device
        self.debug = debug  # Add debug flag
        # move models to device
        self.actor_net.to(device)
        self.critic_net.to(device)
        self.batch_size = batch_size
        self.gamma = 1
        self.lam = 0.97

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
        self.buffer = Buffer()

        self.ppo_update_time = 8
        self.clip_param = 0.2
        self.max_grad_norm = 0.5

        self.entropy_coefficient = 0

        self.actor_optimizer = optim.Adam(
            self.actor_net.parameters(), lr=0.0001, eps=1e-6)
        self.critic_net_optimizer = optim.Adam(
            self.critic_net.parameters(), lr=0.0005, eps=1e-6)

    def choose_action(self, state, mask1, mask2):
        with torch.no_grad():
            probs1 = self.actor_net.getActionn1(state, mask1)
        dist_bin1 = Categorical(probs=probs1)
        ac1 = dist_bin1.sample()
        log_prob1 = dist_bin1.log_prob(ac1)
        job_input = state[:, ac1]
        
        # Carbon-aware mask modification for action2
        # Extract carbon_consideration from job_input
        # The job_input tensor should have the 7 job features, but the indexing might be tricky
        try:
            # If job_input has shape [1, 1, 7], we need job_input[0, 0, 5]
            # If job_input has shape [1, 7], we need job_input[0, 5]
            if len(job_input.shape) == 3 and job_input.shape[2] >= 6:
                carbon_consideration = job_input[0, 0, 5].item()  # 3D tensor case
            elif len(job_input.shape) == 2 and job_input.shape[1] >= 6:
                carbon_consideration = job_input[0, 5].item()  # 2D tensor case
            else:
                # Fallback: if we can't access the carbon consideration, assume 0.0
                carbon_consideration = 0.0
        except (IndexError, RuntimeError) as e:
            # Fallback: if indexing fails for any reason, assume no carbon consideration  
            carbon_consideration = 0.0
        
        # Create carbon-aware mask2: prevent delay for jobs with low carbon consideration
        carbon_aware_mask2 = mask2.clone()
        if carbon_consideration < 0.3:
            # Force action2 = 0 (immediate scheduling) for low carbon consideration jobs
            # Set all delay options to 1 (masked out), keep only action2=0 available
            carbon_aware_mask2[:, 1:] = 1  # Mask out all delay actions (indices 1 and above)
        
        with torch.no_grad():
            probs2 = self.actor_net.getAction2(state, carbon_aware_mask2, job_input)
        dist_bin2 = Categorical(probs=probs2)
        ac2 = dist_bin2.sample()
        log_prob2 = dist_bin2.log_prob(ac2)

        value = self.critic_net(state)
        return ac1, log_prob1, ac2, log_prob2, value, job_input

    def act_job(self, state, mask1, ac1):
        probs1 = self.actor_net.getActionn1(state, mask1)
        dist_bin1 = Categorical(probs=probs1)
        log_prob1 = dist_bin1.log_prob(ac1)
        entropy1 = dist_bin1.entropy()

        return log_prob1, entropy1

    def act_exc(self, state, mask2, job_input, ac2):
        # Carbon-aware mask modification for action2
        # Extract carbon_consideration from job_input
        try:
            # Handle different tensor shapes for batch processing
            if len(job_input.shape) == 3 and job_input.shape[2] >= 6:
                carbon_consideration = job_input[:, 0, 5]  # 3D tensor case, get batch dimension
            elif len(job_input.shape) == 2 and job_input.shape[1] >= 6:
                carbon_consideration = job_input[:, 5]  # 2D tensor case, get batch dimension
            else:
                # Fallback: if we can't access the carbon consideration, assume 0.0 for all in batch
                carbon_consideration = torch.zeros(job_input.shape[0], device=job_input.device)
        except (IndexError, RuntimeError):
            # Fallback: if indexing fails for any reason, assume no carbon consideration  
            carbon_consideration = torch.zeros(job_input.shape[0], device=job_input.device)
        
        # Create carbon-aware mask2: prevent delay for jobs with low carbon consideration
        carbon_aware_mask2 = mask2.clone()
        # Apply mask for each sample in the batch where carbon_consideration < 0.3
        low_carbon_mask = carbon_consideration < 0.3
        if low_carbon_mask.any():
            # For jobs with low carbon consideration, mask out all delay actions
            carbon_aware_mask2[low_carbon_mask, 1:] = 1  # Mask out all delay actions (indices 1 and above)
        
        probs2 = self.actor_net.getAction2(state, carbon_aware_mask2, job_input)
        dist_bin2 = Categorical(probs=probs2)
        log_prob2 = dist_bin2.log_prob(ac2)
        entropy2 = dist_bin2.entropy()
        return log_prob2, entropy2

    def normalize(self, advantages):
        nor_advantages = (advantages - torch.mean(advantages)) / (
                torch.std(advantages) + 1e-9)
        return nor_advantages

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

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        rews = np.append(np.array(self.rewards_seq), last_val)
        values = torch.cat(self.values, dim=0)
        values = values.squeeze(dim=-1)
        vals = np.append(np.array(values.cpu()), last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        adv = self.discount_cumsum(deltas, self.gamma * self.lam)
        # the next line computes rewards-to-go, to be targets for the value function
        ret = self.discount_cumsum(rews, self.gamma)[:-1]
        # ret=adv+vals[:-1]
        return adv, ret

    def storeIntoBuffter(self, reward):
        advantages, returns = self.finish_path(reward)
        returns = returns.tolist()
        advantages = advantages.tolist()

        self.buffer.store_buffer(self.states, self.masks1, self.masks2, self.actions1, self.actions2, self.log_probs1,
                                 self.log_probs2,
                                 returns, advantages, self.job_inputs, len(self.states))

    def compute_value_loss(self, states, returns):
        state_values = self.critic_net(states)
        state_values = torch.squeeze(state_values, dim=1)

        # Calculate value loss using F.mse_loss
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

        log_probs1, entropy1 = self.act_job(states, masks1, actions1)
        log_probs2, entropy2 = self.act_exc(states, masks2, job_input, actions2)
        # Compute the policy loss
        log1 = old_log_probs1 + old_log_probs2
        log2 = log_probs1 + log_probs2
        logratio = log2 - log1
        ratio = torch.exp(logratio)

        surr1 = ratio * advantages
        clip_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
        surr2 = clip_ratio * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))  # MAX->MIN descent
        entropy = (entropy1 + entropy2) / 2
        entropy_loss = torch.mean(entropy)

        total_loss = policy_loss - self.entropy_coefficient * entropy_loss

        # KL divergence approximation KL(old||new) = mean(old_log - new_log)
        kl_mean = torch.mean(log1 - log2).detach()

        return total_loss, policy_loss, entropy_loss, kl_mean

    def train(self):
        states = torch.cat(self.buffer.states, dim=0)
        masks1 = torch.cat(self.buffer.masks1, dim=0)
        masks2 = torch.cat(self.buffer.masks2, dim=0)
        actions1 = torch.cat(self.buffer.actions1, dim=0)
        log_probs1 = torch.cat(self.buffer.log_probs1, dim=0)
        actions2 = torch.cat(self.buffer.actions2, dim=0)
        log_probs2 = torch.cat(self.buffer.log_probs2, dim=0)
        job_inputs = torch.cat(self.buffer.job_inputs, dim=0)
        returns = torch.tensor(self.buffer.Returns)
        advantages = torch.tensor(self.buffer.advantages)
        
        # Store pre-normalized advantages for analysis
        advantages_raw = advantages.clone()
        advantages = self.normalize(advantages)

        # Initialize metrics tracking
        metrics = {
            # Policy optimization metrics
            'policy_loss_clipped': [],
            'policy_loss_unclipped': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
            
            # Critic metrics
            'value_loss': [],
            'explained_variance': [],
            
            # Advantage metrics
            'adv_mean': advantages_raw.mean().item(),
            'adv_std': advantages_raw.std().item(),
            'adv_normalized_mean': advantages.mean().item(),
            'adv_normalized_std': advantages.std().item(),
        }

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer.states))), self.batch_size, False):
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

                # Actor update with detailed metrics
                self.actor_optimizer.zero_grad()
                action_loss, policy_loss, entropy_loss, kl_mean = self.compute_actor_loss(sampled_states, sampled_masks1,
                                                                                sampled_masks2,
                                                                                sampled_actions1, sampled_actions2,
                                                                                sampled_advantages,
                                                                                sampled_log_probs1, sampled_log_probs2,
                                                                                sampled_job_inputs)
                
                # Calculate clipping fraction for this batch
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

                # Critic update with metrics
                self.critic_net_optimizer.zero_grad()
                value_loss = self.compute_value_loss(sampled_states, sampled_returns)
                
                # Calculate explained variance
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

                # Store metrics
                metrics['policy_loss_clipped'].append(action_loss.item())
                metrics['policy_loss_unclipped'].append(policy_loss.item())
                metrics['entropy'].append(entropy_loss.item())
                metrics['approx_kl'].append(kl_mean.item())
                metrics['clip_fraction'].append(clip_fraction)
                metrics['value_loss'].append(value_loss.item())
                metrics['explained_variance'].append(explained_variance)

        # Aggregate metrics
        aggregated_metrics = {}
        for key, values in metrics.items():
            if isinstance(values, list) and len(values) > 0:
                aggregated_metrics[key] = sum(values) / len(values)
            else:
                aggregated_metrics[key] = values

        if self.debug and len(metrics['approx_kl']) > 0:
            print(f"    ↳ Avg KL divergence: {aggregated_metrics['approx_kl']:.6f}")
            
        return aggregated_metrics

    def save_using_model_name(self, model_name_path):
        if not os.path.exists(model_name_path):
            os.makedirs(model_name_path)
        torch.save(self.actor_net.state_dict(), model_name_path + "_actor.pkl")
        torch.save(self.critic_net.state_dict(),
                   model_name_path + "_critic.pkl")

    def load_using_model_name(self, model_name_path):
        self.actor_net.load_state_dict(
            torch.load(model_name_path + "_actor.pkl"))
        self.critic_net.load_state_dict(
            torch.load(model_name_path + "_critic.pkl"))

    def eval_action(self,o,mask1,mask2):
        with torch.no_grad():
            o = o.reshape(1, MAX_QUEUE_SIZE + run_win + green_win, JOB_FEATURES)
            state = torch.FloatTensor(o).to(self.device)
            mask1 = np.array(mask1).reshape(1, MAX_QUEUE_SIZE)
            mask1 = torch.FloatTensor(mask1).to(self.device)
            mask2 = mask2.reshape(1, action2_num)
            mask2 = torch.FloatTensor(mask2).to(self.device)

            probs1 = self.actor_net.getActionn1(state,mask1)
            dist_bin1 = Categorical(probs=probs1)
            ac1 = dist_bin1.sample()
            job_input = state[:, ac1]
            
            # Carbon-aware mask modification for action2
            # Extract carbon_consideration from job_input
            try:
                # Handle different tensor shapes - 3D or 2D
                if len(job_input.shape) == 3 and job_input.shape[2] >= 6:
                    carbon_consideration = job_input[0, 0, 5].item()  # 3D tensor case
                elif len(job_input.shape) == 2 and job_input.shape[1] >= 6:
                    carbon_consideration = job_input[0, 5].item()  # 2D tensor case
                else:
                    # Fallback: if we can't access the carbon consideration, assume 0.0
                    carbon_consideration = 0.0
            except (IndexError, RuntimeError):
                # Fallback: if indexing fails for any reason, assume no carbon consideration  
                carbon_consideration = 0.0
            
            # Create carbon-aware mask2: prevent delay for jobs with low carbon consideration
            carbon_aware_mask2 = mask2.clone()
            if carbon_consideration < 0.3:
                # Force action2 = 0 (immediate scheduling) for low carbon consideration jobs
                # Set all delay options to 1 (masked out), keep only action2=0 available
                carbon_aware_mask2[:, 1:] = 1  # Mask out all delay actions (indices 1 and above)
            
            probs2 = self.actor_net.getAction2(state, carbon_aware_mask2, job_input)
            dist_bin2 = Categorical(probs=probs2)
            ac2 = dist_bin2.sample()

        return ac1, ac2

class PPOModelWrapper:
    """
    Wrapper class to make PPO model compatible with validation system.
    The validation system expects a model with an eval_action method.
    """
    def __init__(self, ppo_model):
        self.ppo_model = ppo_model
        
    def eval_action(self, o, mask1, mask2):
        """
        Wrapper for eval_action to match validation interface
        """
        return self.ppo_model.eval_action(o, mask1, mask2)

def calculate_epoch_score(ppo_model, env, experiment_dir, epoch, workload, debug=False):
    """
    Calculate score for the current epoch by running validation and score calculation.
    
    Args:
        ppo_model: Trained PPO model
        env: Environment instance
        experiment_dir: Experiment directory path
        epoch: Current epoch number
        workload: Workload name
        debug: Debug flag
    
    Returns:
        float: Calculated score (or None if calculation fails)
    """
    if not (SCORE_CALCULATION_AVAILABLE and VALIDATION_AVAILABLE):
        if debug:
            print(f"  Score calculation skipped (dependencies not available)")
        return None
    
    try:
        if debug:
            print(f"  Running validation for score calculation...")
        
        # Wrap the PPO model for compatibility with validation system
        wrapped_model = PPOModelWrapper(ppo_model)
        
        # Initialize enhanced tracker for validation
        tracker = EnhancedJobTracker()
        
        # Run multiple validation episodes to collect more data for better analysis
        # Read validation episodes from config
        config = configparser.ConfigParser()
        config.read('configFile/config.ini')
        total_episodes = int(config.get('training parameters', 'validation_episodes', fallback=5))
        episode_results = []
        
        if debug:
            print(f"  Running {total_episodes} validation episodes for score calculation...")
        
        for episode in range(total_episodes):
            env.reset()
            total_reward, green_reward, jobs_completed = run_enhanced_marl_validation(
                wrapped_model, env, tracker, episode + 1
            )
            episode_results.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'green_reward': green_reward,
                'jobs_completed': jobs_completed
            })
            
            if debug:
                print(f"    Episode {episode + 1}: {len([j for j in tracker.jobs_data if j['scheduled']])} jobs scheduled")
        
        # Check if we have enough data for score calculation
        # Scale minimum required data based on number of episodes
        min_jobs_required = max(20, total_episodes * 10)  # At least 20 jobs, or 10 per episode
        if len(tracker.jobs_data) < min_jobs_required or len(tracker.carbon_window_data) < min_jobs_required:
            if debug:
                print(f"  Insufficient data for score calculation (jobs: {len(tracker.jobs_data)}, windows: {len(tracker.carbon_window_data)})")
                print(f"  Required: {min_jobs_required} jobs minimum (based on {total_episodes} episodes)")
                print(f"  Consider running more episodes or longer episodes")
            return None
        
        # Prepare data for score calculation
        jobs_df = pd.DataFrame(tracker.jobs_data)
        scheduled_jobs = jobs_df[jobs_df['scheduled'] == True]
        
        if len(scheduled_jobs) < 5:
            if debug:
                print(f"  Insufficient scheduled jobs for score calculation ({len(scheduled_jobs)})")
            return None
        
        # Add carbon window data
        window_df = pd.DataFrame(tracker.carbon_window_data)
        if len(window_df) > 0:
            # Merge carbon window averages with job data
            window_summary = window_df.groupby('job_id').agg({
                'carbon_intensity_avg': 'first',
                'carbon_intensity_min': 'first', 
                'carbon_intensity_max': 'first'
            }).reset_index()
            
            scheduled_jobs = scheduled_jobs.merge(window_summary, on='job_id', how='left')
            scheduled_jobs['carbon_intensity_baseline'] = scheduled_jobs['carbon_intensity_avg'].fillna(350.0)
        else:
            scheduled_jobs['carbon_intensity_baseline'] = 350.0  # Default baseline
        
        # Create interaction terms for regression
        scheduled_jobs['runtime_x_processors'] = scheduled_jobs['request_time'] * scheduled_jobs['request_processors']
        scheduled_jobs['carbon_x_runtime_x_processors'] = (
            scheduled_jobs['carbon_consideration'] * 
            scheduled_jobs['request_time'] * 
            scheduled_jobs['request_processors']
        )
        
        # Check for required columns
        required_cols = ['carbon_consideration', 'carbon_emissions', 'wait_time', 
                        'queue_length_at_submission', 'carbon_intensity_baseline']
        missing_cols = [col for col in required_cols if col not in scheduled_jobs.columns]
        if missing_cols:
            if debug:
                print(f"  Missing required columns for score calculation: {missing_cols}")
            return None
        
        # Run regressions
        carbon_model, _, _ = run_carbon_emissions_regression(scheduled_jobs)
        wait_model, _, _ = run_wait_time_regression(scheduled_jobs) 
        
        # Calculate score
        score = calculate_score(carbon_model, wait_model)
        
        # Save score to epoch-specific file
        score_file = f"{experiment_dir}/score_epoch_{epoch}.txt"
        with open(score_file, 'w') as f:
            f.write(f"Epoch {epoch} Score Calculation\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Score: {score:.6f}\n")
            f.write(f"Episodes run: {total_episodes}\n")
            f.write(f"Jobs analyzed: {len(scheduled_jobs)}\n")
            f.write(f"Carbon windows captured: {len(tracker.carbon_window_data)}\n\n")
            
            # Add episode-level statistics
            if episode_results:
                avg_total_reward = sum(ep['total_reward'] for ep in episode_results) / len(episode_results)
                avg_green_reward = sum(ep['green_reward'] for ep in episode_results) / len(episode_results)
                f.write(f"Episode Statistics:\n")
                f.write(f"Average total reward: {avg_total_reward:.2f}\n")
                f.write(f"Average green reward: {avg_green_reward:.4f}\n\n")
            
            f.write("Carbon Emissions Regression Summary:\n")
            f.write(f"R-squared: {carbon_model.rsquared:.4f}\n")
            f.write(f"Carbon consideration coef: {carbon_model.params['carbon_consideration']:.6f}\n\n")
            
            f.write("Wait Time Regression Summary:\n")
            f.write(f"R-squared: {wait_model.rsquared:.4f}\n")
            f.write(f"Carbon consideration coef: {wait_model.params['carbon_consideration']:.6f}\n")
        
        # Also append to main scores CSV
        scores_csv = f"{experiment_dir}/epoch_scores.csv"
        file_exists = os.path.exists(scores_csv)
        
        with open(scores_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'score', 'episodes_run', 'carbon_r2', 'wait_r2', 'carbon_coef', 'wait_coef', 'jobs_analyzed'])
            
            writer.writerow([
                epoch,
                score,
                total_episodes,
                carbon_model.rsquared,
                wait_model.rsquared,
                carbon_model.params['carbon_consideration'],
                wait_model.params['carbon_consideration'],
                len(scheduled_jobs)
            ])
        
        if debug:
            print(f"  Score calculated: {score:.4f} (saved to score_epoch_{epoch}.txt)")
        
        return score
        
    except Exception as e:
        if debug:
            print(f"  Score calculation failed: {str(e)}")
        return None

def run_validation_episodes(ppo_model, validation_env, validation_episodes, debug=False):
    """
    Run validation episodes on a separate environment with fixed seed.
    
    Args:
        ppo_model: The trained PPO model
        validation_env: Validation environment with fixed seed
        validation_episodes: Number of validation episodes to run
        debug: Whether to print debug information
        
    Returns:
        dict: Validation metrics including average rewards and episode statistics
    """
    validation_rewards = []
    validation_green_rewards = []
    validation_wait_rewards = []
    validation_episode_lengths = []
    validation_jobs_completed = []
    
    if debug:
        print(f"    Running {validation_episodes} validation episodes...")
    
    for episode in range(validation_episodes):
        # Reset validation environment
        o, r, d = validation_env.reset(), 0, False
        ep_ret, ep_len, green_reward, wait_reward = 0, 0, 0, 0
        running_num = 0
        jobs_completed = 0
        
        while not d:
            # Build action masks (same logic as training)
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                job_slice = o[i:i + JOB_FEATURES]
                
                # Check for padding patterns
                padding_pattern1 = [0, 1, 1, 1, 1, 0.5, 0]
                padding_pattern2 = [1] * JOB_FEATURES
                
                if (len(job_slice) == len(padding_pattern1) and 
                    all(abs(job_slice[j] - padding_pattern1[j]) < 1e-6 for j in range(len(job_slice)))):
                    lst.append(1)  # Mask out (invalid)
                elif (len(job_slice) == len(padding_pattern2) and 
                      all(abs(job_slice[j] - padding_pattern2[j]) < 1e-6 for j in range(len(job_slice)))):
                    lst.append(1)  # Mask out (invalid)
                else:
                    lst.append(0)  # Valid job
            
            mask2 = np.zeros(action2_num, dtype=int)
            if running_num < delayMaxJobNum:
                mask2[running_num + 1 : delayMaxJobNum + 1] = 1
            
            # Get action from policy (deterministic evaluation)
            action1, action2 = ppo_model.eval_action(o, lst, mask2)
            
            # Step environment
            o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = \
                validation_env.step(action1.item(), action2.item())
            
            # Accumulate rewards
            ep_ret += r + greenRwd  # Total reward (wait + green)
            wait_reward += r
            green_reward += greenRwd
            ep_len += 1
            
            # Count completed jobs (when episode ends)
            if d:
                # Rough estimate: count the valid jobs that were processed
                jobs_completed = ep_len  # This is an approximation
        
        # Store episode metrics
        validation_rewards.append(ep_ret)
        validation_green_rewards.append(green_reward)
        validation_wait_rewards.append(wait_reward)
        validation_episode_lengths.append(ep_len)
        validation_jobs_completed.append(jobs_completed)
        
        if debug:
            print(f"      Episode {episode + 1}: {ep_len} steps, reward: {ep_ret:.4f} (green: {green_reward:.4f}, wait: {wait_reward:.4f})")
    
    # Calculate validation metrics
    validation_metrics = {
        'avg_total_reward': sum(validation_rewards) / len(validation_rewards),
        'avg_green_reward': sum(validation_green_rewards) / len(validation_green_rewards),
        'avg_wait_reward': sum(validation_wait_rewards) / len(validation_wait_rewards),
        'avg_episode_length': sum(validation_episode_lengths) / len(validation_episode_lengths),
        'avg_jobs_completed': sum(validation_jobs_completed) / len(validation_jobs_completed),
        'std_total_reward': np.std(validation_rewards),
        'median_total_reward': np.median(validation_rewards),
        'episodes_run': validation_episodes
    }
    
    return validation_metrics

def setup_experiment_directory(workload, experiment_name, description):
    """
    Setup the experiment directory structure:
    workload/
      └── MARL_experiment_name/
          ├── description.txt
          ├── config.ini (copy)
          ├── checkpoints/
          │   ├── epoch_5/
          │   ├── epoch_10/
          │   └── ...
          └── final/
    """
    
    # Create experiment directory
    experiment_dir = f"{workload}/MARL_{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{experiment_dir}/final", exist_ok=True)
    
    # Save description
    if description:
        with open(f"{experiment_dir}/description.txt", 'w') as f:
            f.write(f"Experiment: MARL_{experiment_name}\n")
            f.write(f"Workload: {workload}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Description: {description}\n")
    
    # Save complete configuration snapshot
    with open(f"{experiment_dir}/config_snapshot.ini", 'w') as f:
        # Read current config
        config = configparser.ConfigParser()
        config.read('configFile/config.ini')
        
        # Write header comment
        f.write(f"# Configuration snapshot for experiment MARL_{experiment_name}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# All constants and parameters used for this experiment\n\n")
        
        # Write all config sections
        for section in config.sections():
            f.write(f"[{section}]\n")
            for key, value in config.items(section):
                f.write(f"{key} = {value}\n")
            f.write("\n")
    
    return experiment_dir
        
def train(workload, backfill, debug=False, experiment_name="", description="", no_score=False, validate_every_n_epochs=False):
    print("Training called")
    # ------------------------------------------------------------------
    # 1. Experiment-wide hyper-parameters & environment construction
    # ------------------------------------------------------------------
    # Read training parameters from config
    config = configparser.ConfigParser()
    config.read('configFile/config.ini')
    
    seed       = int(config.get('training parameters', 'seed'))
    epochs     = int(config.get('training parameters', 'epochs'))
    traj_num   = int(config.get('training parameters', 'traj_num'))
    
    # Validation parameters
    validation_episodes = int(config.get('training parameters', 'validation_episodes', fallback=5))
    validation_seed = int(config.get('training parameters', 'validation_seed', fallback=42))
    validation_interval = int(config.get('training parameters', 'validation_interval', fallback=5))
    
    # Load delay time list for debug statistics
    delaytimelist_str = config.get('GAS-MARL setting', 'delaytimelist')
    # Parse the list string to actual list (format: [1100,2200,5400,10800,21600,43200,86400])
    delaytimelist = eval(delaytimelist_str)
    
    # Setup experiment directory structure
    experiment_dir = setup_experiment_directory(workload, experiment_name, description)
    print(f"Experiment directory: {experiment_dir}")
    
    # Always print training parameters (not just in debug mode)
    print(f"Training parameters:")
    print(f"  Workload: {workload}")
    print(f"  Backfill: {backfill}")
    print(f"  Epochs: {epochs}")
    print(f"  Trajectories per epoch: {traj_num}")
    print(f"  Seed: {seed}")
    print(f"  Eta: {float(config.get('GAS-MARL setting', 'eta', fallback=0.5))}")
    
    # Print other important configuration variables
    print(f"Configuration:")
    print(f"  Max queue size: {config.get('GAS-MARL setting', 'max_queue_size')}")
    print(f"  Run window: {config.get('GAS-MARL setting', 'run_win')}")
    print(f"  Green window: {config.get('GAS-MARL setting', 'green_win')}")
    print(f"  Delay max jobs: {config.get('GAS-MARL setting', 'delaymaxjobnum')}")
    print(f"  Use constant power: {config.get('power setting', 'use_constant_power')}")
    if config.getboolean('power setting', 'use_constant_power', fallback=False):
        print(f"  Constant power per processor: {config.get('power setting', 'constant_power_per_processor')}")
    print(f"  Use dynamic carbon window: {config.get('carbon setting', 'use_dynamic_window')}")
    print(f"  Reward function: {config.get('reward_config', 'reward_function', fallback='legacy')}")
    print(f"  Carbon year: {config.get('general setting', 'carbon_year')}")
    
    env = HPCEnv(backfill=backfill, debug=debug)  # custom cluster-scheduling env
    env.seed(seed)                  # reproducible RNG for env
    current_dir   = os.getcwd()
    workload_file = os.path.join(current_dir, "data", f"{workload}.swf")
    
    # Check if the specified workload file exists
    if not os.path.exists(workload_file):
        print(f"ERROR: Workload file not found: {workload_file}")
        print(f"Available workload files in data/:")
        data_dir = os.path.join(current_dir, "data")
        if os.path.exists(data_dir):
            swf_files = [f for f in os.listdir(data_dir) if f.endswith('.swf')]
            for swf_file in sorted(swf_files):
                print(f"  {swf_file}")
        exit(1)
    
    print(f"Using workload: {workload_file}")
    env.my_init(workload_file=workload_file)  # load the SWF trace
    
    # Create separate validation environment with fixed seed for consistent evaluation
    validation_env = None
    if validate_every_n_epochs:
        validation_env = HPCEnv(backfill=backfill, debug=False)  # No debug for validation
        validation_env.seed(validation_seed)  # Fixed seed for reproducible validation
        validation_env.my_init(workload_file=workload_file)
        print(f"Validation environment created with seed: {validation_seed}")
        print(f"Validation will run every {validation_interval} epochs with {validation_episodes} episodes")
    
    if debug:
        print(f"DEBUG: Workload loaded successfully")
        print(f"  Number of jobs: {env.loads.size()}")
        print(f"  Max execution time: {env.loads.max_exec_time}")
        print(f"  Max processors: {env.loads.max_procs}")
        # Check first job's carbon consideration
        if env.loads.size() > 0:
            first_job = env.loads[0]
            print(f"  First job carbon consideration: {first_job.carbon_consideration}")
            print(f"  Job features: {JOB_FEATURES}")
    
    # ------------------------------------------------------------------
    # 2. Device selection (GPU if available)
    # ------------------------------------------------------------------
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    
    # ------------------------------------------------------------------
    # 3. PPO agent creation
    # ------------------------------------------------------------------
    # Read eta from config
    eta = float(config.get('GAS-MARL setting', 'eta', fallback=0.5))
    if debug:
        print(f"  Using eta: {eta}")
        
    inputNum_size    = [MAX_QUEUE_SIZE, run_win, green_win]       # grid dims
    featureNum_size  = [JOB_FEATURES,   RUN_FEATURE, GREEN_FEATURE]
    
    ppo = PPO(
        batch_size      = 256,
        inputNum_size   = inputNum_size,
        featureNum_size = featureNum_size,
        device          = device,
        debug           = debug
    )
    
    # ------------------------------------------------------------------
    # 4. Outer training loop  (epochs)
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        if debug:
            print(f"\nDEBUG: Starting epoch {epoch + 1}/{epochs}")
        # Reset env and bookkeeping for the *first* trajectory of the epoch
        o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = (
            env.reset(), 0, False, 0, 0, 0, 0, 0, 0
        )
        running_num  = 0   # jobs currently running
        t            = 0   # trajectory counter within this epoch
        epoch_reward = 0
        green_reward = 0
        wait_reward  = 0
        
        # --------------------------------------------------------------
        # 5. Collect trajectories until we hit traj_num
        # --------------------------------------------------------------
        trajectory_stats = {"valid_jobs": [], "actions": [], "rewards": []}
        
        # Enhanced action statistics for debug mode
        if debug:
            epoch_action_stats = {
                "action1_choices": [],      # job selection indices
                "action2_choices": [],      # delay choices
                "episode_lengths": [],     # steps per episode
                "backfill_actions": [],    # action2 > 0 indicates delay/backfill
                "delay_times": [],         # actual delay times in seconds
                "valid_jobs_per_step": []  # number of valid jobs per step
            }
        
        while True:
            # ----------------------------------------------------------
            # 5-a. Build action masks
            #      mask1 → which queue entries are *valid* jobs
            #      mask2 → constraints on action2 (delay index)
            # ----------------------------------------------------------
            lst = []
            for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
                job_slice = o[i:i + JOB_FEATURES]
                
                # Check for padding patterns (mask out with 1)
                # Pattern 1: [0, 1, 1, 1, 1, 0.5, 0] - job queue padding from HPCSimPickJobs.py line 377
                padding_pattern1 = [0, 1, 1, 1, 1, 0.5, 0]
                # Pattern 2: [1, 1, 1, 1, 1, 1, 1] - all ones padding
                padding_pattern2 = [1] * JOB_FEATURES
                
                if (len(job_slice) == len(padding_pattern1) and 
                    all(abs(job_slice[j] - padding_pattern1[j]) < 1e-6 for j in range(len(job_slice)))):
                    lst.append(1)  # Mask out (invalid)
                elif (len(job_slice) == len(padding_pattern2) and 
                      all(abs(job_slice[j] - padding_pattern2[j]) < 1e-6 for j in range(len(job_slice)))):
                    lst.append(1)  # Mask out (invalid)
                else:
                    lst.append(0)  # Valid job
            
            # Collect stats for aggregated debugging
            valid_jobs = lst.count(0)
            trajectory_stats["valid_jobs"].append(valid_jobs)
            
            mask2 = np.zeros(action2_num, dtype=int)
            if running_num < delayMaxJobNum:                 # cannot delay past running capacity
                mask2[running_num + 1 : delayMaxJobNum + 1] = 1
            
            # ----------------------------------------------------------
            # 5-b. State → tensor; ask PPO for an action
            # ----------------------------------------------------------
            with torch.no_grad():                            # inference only
                o_reshaped = o.reshape(1, MAX_QUEUE_SIZE + run_win + green_win, JOB_FEATURES)
                state  = torch.FloatTensor(o_reshaped).to(device)
                mask1T = torch.FloatTensor(np.array(lst).reshape(1, MAX_QUEUE_SIZE)).to(device)
                mask2T = torch.FloatTensor(mask2.reshape(1, action2_num)).to(device)
                
                action1, log_prob1, action2, log_prob2, value, job_input = \
                    ppo.choose_action(state, mask1T, mask2T)
            
            # Store transition in PPO's rollout buffer
            ppo.remember(
                state, value, log_prob1, log_prob2, action1,
                action2, greenRwd, mask1T, mask2T, device, job_input
            )
            
            # Collect action stats
            trajectory_stats["actions"].append((action1.item(), action2.item()))
            
            # Enhanced debug statistics collection
            if debug:
                epoch_action_stats["action1_choices"].append(action1.item())
                epoch_action_stats["action2_choices"].append(action2.item())
                epoch_action_stats["valid_jobs_per_step"].append(valid_jobs)
                
                # Check if this is a backfill action (delay > 0)
                if action2.item() > 0:
                    epoch_action_stats["backfill_actions"].append(action2.item())
                    # Convert delay index to actual time using delaytimelist
                    delay_index = action2.item() - 1  # action2 is 1-indexed for delays
                    if 0 <= delay_index < len(delaytimelist):
                        epoch_action_stats["delay_times"].append(delaytimelist[delay_index])
                    else:
                        epoch_action_stats["delay_times"].append(0)
            
            # ----------------------------------------------------------
            # 5-c. Step the environment
            # ----------------------------------------------------------
            o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = \
                env.step(action1.item(), action2.item())
                
            # Collect reward stats
            trajectory_stats["rewards"].append((r, greenRwd))
            
            # Trajectory-level bookkeeping
            ep_ret   += r
            ep_len   += 1
            show_ret += r2      # additional metric for reporting
            sjf      += sjf_t   # schedule fairness proxy
            f1       += f1_t    # some QoS metric
            
            # Reward components
            green_reward += greenRwd   # energy-aware part
            wait_reward  += r          # waiting-time part
            epoch_reward += eta * r + greenRwd

            
            
            # ----------------------------------------------------------
            # 5-d. If episode ends → push terminal reward, maybe finish epoch
            # ----------------------------------------------------------
            if d:
                # Store episode length for debug stats
                if debug:
                    epoch_action_stats["episode_lengths"].append(ep_len)
                
                # Print aggregated trajectory summary for debugging
                if debug and t < 3:
                    avg_valid_jobs = sum(trajectory_stats["valid_jobs"]) / len(trajectory_stats["valid_jobs"]) if trajectory_stats["valid_jobs"] else 0
                    total_rewards = sum([r[0] + r[1] for r in trajectory_stats["rewards"]])
                    print(f"  Episode {t + 1}: {ep_len} steps, {ep_ret:.4f} return, avg {avg_valid_jobs:.1f} valid jobs, total reward: {total_rewards:.4f}")
                    
                t += 1                                     # finished a trajectory
                ppo.storeIntoBuffter(eta * r + greenRwd)   # final R for GAE
                ppo.clear_memory()                         # reset per-traj state
                
                # reset env for next trajectory
                o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = (
                    env.reset(), 0, False, 0, 0, 0, 0, 0, 0
                )
                running_num = 0
                
                # Reset trajectory stats for next episode
                trajectory_stats = {"valid_jobs": [], "actions": [], "rewards": []}
                
                if t >= traj_num:     # collected enough rollouts for this epoch
                    if debug:
                        print(f"  Epoch {epoch + 1} data collection complete ({t} trajectories)")
                    break
        
        # --------------------------------------------------------------
        # 6. Policy / value-function update after traj_num rollouts
        # --------------------------------------------------------------
        if debug:
            print(f"  Training policy for epoch {epoch + 1}...")
            
        training_metrics = ppo.train()
        
        # --------------------------------------------------------------
        # 7. Enhanced metrics collection for logging
        # --------------------------------------------------------------
        avg_epoch_reward = epoch_reward / traj_num
        avg_green_reward = green_reward / traj_num
        avg_wait_reward = wait_reward / traj_num
        
        # Collect episode return statistics from trajectory rewards
        episode_returns = []
        carbon_costs = []
        latencies = []
        
        # Calculate episode-level metrics from collected rewards
        for trajectory_rewards in trajectory_rewards_list if 'trajectory_rewards_list' in locals() else []:
            episode_total = sum([r[0] + r[1] for r in trajectory_rewards])
            episode_returns.append(episode_total)
            carbon_costs.append(sum([r[1] for r in trajectory_rewards]))
            latencies.append(sum([r[0] for r in trajectory_rewards]))
        
        # If no trajectory-level data is available, use averaged values
        if not episode_returns:
            episode_returns = [avg_epoch_reward] * traj_num
            carbon_costs = [avg_green_reward] * traj_num  
            latencies = [avg_wait_reward] * traj_num
        
        episode_return_mean = sum(episode_returns) / len(episode_returns) if episode_returns else avg_epoch_reward
        episode_return_median = sorted(episode_returns)[len(episode_returns)//2] if episode_returns else avg_epoch_reward
        carbon_cost_mean = sum(carbon_costs) / len(carbon_costs) if carbon_costs else avg_green_reward
        latency_mean = sum(latencies) / len(latencies) if latencies else avg_wait_reward
        
        # Run validation if enabled and at the right interval
        validation_metrics = None
        if validate_every_n_epochs and (epoch + 1) % validation_interval == 0:
            if debug:
                print(f"  → Running validation for epoch {epoch + 1}...")
            validation_metrics = run_validation_episodes(ppo, validation_env, validation_episodes, debug)
        
        # Save to experiment directory with enhanced metrics
        csv_file = f"{experiment_dir}/training_results.csv"
        # Write header if file doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="") as file:
                csv.writer(file).writerow([
                    "epoch", "episode_return_mean", "episode_return_median", "avg_epoch_reward", 
                    "avg_green_reward", "avg_wait_reward", "carbon_cost_mean", "latency_mean",
                    "policy_loss_clipped", "policy_loss_unclipped", "entropy", "approx_kl", 
                    "clip_fraction", "value_loss", "explained_variance", "adv_mean", "adv_std",
                    "validation_avg_total_reward", "validation_avg_green_reward", "validation_avg_wait_reward",
                    "validation_std_total_reward", "validation_median_total_reward", "validation_avg_episode_length"
                ])
        
        with open(csv_file, mode="a", newline="") as file:
            csv.writer(file).writerow([
                epoch + 1,
                episode_return_mean,
                episode_return_median,
                avg_epoch_reward,
                avg_green_reward,
                avg_wait_reward,
                carbon_cost_mean,
                latency_mean,
                training_metrics.get('policy_loss_clipped', 0) if training_metrics else 0,
                training_metrics.get('policy_loss_unclipped', 0) if training_metrics else 0,
                training_metrics.get('entropy', 0) if training_metrics else 0,
                training_metrics.get('approx_kl', 0) if training_metrics else 0,
                training_metrics.get('clip_fraction', 0) if training_metrics else 0,
                training_metrics.get('value_loss', 0) if training_metrics else 0,
                training_metrics.get('explained_variance', 0) if training_metrics else 0,
                training_metrics.get('adv_mean', 0) if training_metrics else 0,
                training_metrics.get('adv_std', 0) if training_metrics else 0,
                validation_metrics.get('avg_total_reward', '') if validation_metrics else '',
                validation_metrics.get('avg_green_reward', '') if validation_metrics else '',
                validation_metrics.get('avg_wait_reward', '') if validation_metrics else '',
                validation_metrics.get('std_total_reward', '') if validation_metrics else '',
                validation_metrics.get('median_total_reward', '') if validation_metrics else '',
                validation_metrics.get('avg_episode_length', '') if validation_metrics else ''
            ])
        
        # ══════════════════════════════════════════════════════════════
        # COMPREHENSIVE EPOCH METRICS SUMMARY
        # ══════════════════════════════════════════════════════════════
        print(f"\n🎯 EPOCH {epoch + 1:3d}/{epochs} METRICS")
        print(f"{'='*60}")
        
        # Top-line Performance
        print(f"📊 TOP-LINE PERFORMANCE:")
        print(f"   Episode Return Mean:   {episode_return_mean:8.4f}")
        print(f"   Episode Return Median: {episode_return_median:8.4f}")
        print(f"   Running Average:       {avg_epoch_reward:8.4f}")
        
        # Reward Decomposition  
        print(f"🎨 REWARD DECOMPOSITION:")
        print(f"   Carbon Cost Mean:      {carbon_cost_mean:8.4f}")
        print(f"   Latency Mean:          {latency_mean:8.4f}")
        print(f"   Green Reward:          {avg_green_reward:8.4f}")
        print(f"   Wait Reward:           {avg_wait_reward:8.4f}")
        
        # Policy Optimization
        print(f"🔧 POLICY OPTIMIZATION:")
        if training_metrics:
            print(f"   Policy Loss (Clipped): {training_metrics.get('policy_loss_clipped', 0):8.6f}")
            print(f"   Policy Loss (Raw):     {training_metrics.get('policy_loss_unclipped', 0):8.6f}")
            print(f"   Entropy:               {training_metrics.get('entropy', 0):8.6f}")
            print(f"   Approx KL:             {training_metrics.get('approx_kl', 0):8.6f}")
            print(f"   Clip Fraction:         {training_metrics.get('clip_fraction', 0):8.6f}")
        else:
            print(f"   [Training metrics not available]")
        
        # Critic Health
        print(f"💡 CRITIC HEALTH:")
        if training_metrics:
            print(f"   Value Loss:            {training_metrics.get('value_loss', 0):8.6f}")
            print(f"   Explained Variance:    {training_metrics.get('explained_variance', 0):8.6f}")
        else:
            print(f"   [Critic metrics not available]")
        
        # Advantages
        print(f"⚡ ADVANTAGES:")
        if training_metrics:
            print(f"   Advantage Mean (Raw):  {training_metrics.get('adv_mean', 0):8.6f}")
            print(f"   Advantage Std (Raw):   {training_metrics.get('adv_std', 0):8.6f}")
            print(f"   Advantage Mean (Norm): {training_metrics.get('adv_normalized_mean', 0):8.6f}")
            print(f"   Advantage Std (Norm):  {training_metrics.get('adv_normalized_std', 0):8.6f}")
        else:
            print(f"   [Advantage metrics not available]")
        
        print(f"{'='*60}")
        
        # Legacy compact summary for quick reference
        print(f"Epoch {epoch + 1:3d} | Ret: {episode_return_mean:7.3f} | Green: {avg_green_reward:7.3f} | Wait: {avg_wait_reward:7.3f}")
        
        # Display validation results if available
        if validation_metrics:
            print(f"📋 VALIDATION RESULTS (Epoch {epoch + 1}):")
            print(f"   Avg Total Reward:    {validation_metrics['avg_total_reward']:8.4f} ± {validation_metrics['std_total_reward']:6.4f}")
            print(f"   Avg Green Reward:    {validation_metrics['avg_green_reward']:8.4f}")
            print(f"   Avg Wait Reward:     {validation_metrics['avg_wait_reward']:8.4f}")
            print(f"   Median Total Reward: {validation_metrics['median_total_reward']:8.4f}")
            print(f"   Avg Episode Length:  {validation_metrics['avg_episode_length']:8.1f} steps")
        
        # Enhanced debug statistics after each epoch
        if debug and 'epoch_action_stats' in locals():
            print(f"  ═══ Epoch {epoch + 1} Action Statistics ═══")
            
            # Episode statistics
            if epoch_action_stats["episode_lengths"]:
                avg_episode_length = sum(epoch_action_stats["episode_lengths"]) / len(epoch_action_stats["episode_lengths"])
                print(f"  📊 Episodes: {len(epoch_action_stats['episode_lengths'])} total, avg {avg_episode_length:.1f} steps/episode")
            
            # Job selection (action1) statistics
            if epoch_action_stats["action1_choices"]:
                action1_distribution = {}
                for action in epoch_action_stats["action1_choices"]:
                    action1_distribution[action] = action1_distribution.get(action, 0) + 1
                top_job_choices = sorted(action1_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  🎯 Job Selection: Top job indices chosen: {top_job_choices}")
                print(f"      Avg job index: {sum(epoch_action_stats['action1_choices']) / len(epoch_action_stats['action1_choices']):.1f}")
            
            # Delay/backfill (action2) statistics
            total_actions = len(epoch_action_stats["action2_choices"])
            immediate_schedules = epoch_action_stats["action2_choices"].count(0)
            backfill_count = total_actions - immediate_schedules
            
            if total_actions > 0:
                print(f"  ⏰ Scheduling: {immediate_schedules}/{total_actions} ({immediate_schedules/total_actions*100:.1f}%) immediate")
                print(f"      Backfill: {backfill_count}/{total_actions} ({backfill_count/total_actions*100:.1f}%) delayed")
                
                if epoch_action_stats["delay_times"]:
                    avg_delay = sum(epoch_action_stats["delay_times"]) / len(epoch_action_stats["delay_times"])
                    max_delay = max(epoch_action_stats["delay_times"])
                    print(f"      Delays: avg {avg_delay/3600:.1f}h, max {max_delay/3600:.1f}h ({len(epoch_action_stats['delay_times'])} delays)")
                    
                    # Delay distribution
                    delay_distribution = {}
                    for delay in epoch_action_stats["delay_times"]:
                        delay_hours = round(delay / 3600, 1)
                        delay_distribution[delay_hours] = delay_distribution.get(delay_hours, 0) + 1
                    top_delays = sorted(delay_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
                    print(f"      Top delay times: {top_delays} (hours)")
            
            # Valid jobs statistics
            if epoch_action_stats["valid_jobs_per_step"]:
                avg_valid_jobs = sum(epoch_action_stats["valid_jobs_per_step"]) / len(epoch_action_stats["valid_jobs_per_step"])
                min_valid = min(epoch_action_stats["valid_jobs_per_step"])
                max_valid = max(epoch_action_stats["valid_jobs_per_step"])
                print(f"  📋 Queue Load: avg {avg_valid_jobs:.1f} valid jobs/step (range: {min_valid}-{max_valid})")
            
            print(f"  ══════════════════════════════════════")
        
        # Save model weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{experiment_dir}/checkpoints/epoch_{epoch + 1}/"
            ppo.save_using_model_name(checkpoint_path)
            if debug:
                print(f"  → Checkpoint saved: epoch_{epoch + 1}/")
            
            # Calculate score every 5th epoch
            if not no_score:
                if debug:
                    print(f"  → Calculating score for epoch {epoch + 1}...")
                score = calculate_epoch_score(ppo, env, experiment_dir, epoch + 1, workload, debug)
                if score is not None:
                    print(f"Epoch {epoch + 1:3d} Score: {score:7.4f}")
                else:
                    print(f"Epoch {epoch + 1:3d} Score: calculation skipped")
        
        # clear buffers so next epoch starts fresh
        ppo.buffer.clear_buffer()
    
    # ------------------------------------------------------------------
    # 8. Persist the trained model
    # ------------------------------------------------------------------
    final_model_path = f"{experiment_dir}/final/"
    ppo.save_using_model_name(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Also save in the legacy location for backward compatibility
    legacy_path = f"{workload}/MARL/"
    ppo.save_using_model_name(legacy_path)
    print(f"Legacy model saved to: {legacy_path}")
    
    print(f"Training completed!")
    return experiment_dir




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='lublin_256', help='Workload dataset to use')
    parser.add_argument('--backfill', type=int, default=0, help='Backfill strategy (0/1)')
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    parser.add_argument('--name', type=str, required=True, help='Experiment name (e.g., ED12)')
    parser.add_argument('--description', type=str, default='', help='Description of the experiment')
    parser.add_argument('--no-score', action='store_true', help='Disable score calculation during training')
    parser.add_argument('--validate', action='store_true', help='Enable validation every N epochs on a fixed validation set')
    args = parser.parse_args()
    train(args.workload, args.backfill, args.debug, args.name, args.description, args.no_score, args.validate)