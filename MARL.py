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
                 device='cpu'):
        super(PPO, self).__init__()
        self.num_inputs1 = inputNum_size[0]
        self.num_inputs2 = inputNum_size[1]
        self.num_inputs3 = inputNum_size[2]

        self.featureNum1 = featureNum_size[0]
        self.featureNum2 = featureNum_size[1]
        self.featureNum3 = featureNum_size[2]

        self.device = device
        self.actor_net = ActorNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3).to(self.device)
        self.critic_net = CriticNet(
            self.num_inputs1, self.featureNum1, self.num_inputs2, self.featureNum2, self.num_inputs3,
            self.featureNum3).to(self.device)
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
        with torch.no_grad():
            probs2 = self.actor_net.getAction2(state, mask2, job_input)
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
        probs2 = self.actor_net.getAction2(state, mask2, job_input)
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

        return total_loss, policy_loss, entropy_loss

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
        advantages = self.normalize(advantages)

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

                self.actor_optimizer.zero_grad()
                action_loss, polic_loss, entropy_loss = self.compute_actor_loss(sampled_states, sampled_masks1,
                                                                                sampled_masks2,
                                                                                sampled_actions1, sampled_actions2,
                                                                                sampled_advantages,
                                                                                sampled_log_probs1, sampled_log_probs2,
                                                                                sampled_job_inputs)
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_net_optimizer.zero_grad()
                value_loss = self.compute_value_loss(sampled_states, sampled_returns)
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

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
            probs2 = self.actor_net.getAction2(state, mask2, job_input)
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
        
        # Run a single validation episode to collect data
        env.reset()
        total_reward, green_reward, jobs_completed = run_enhanced_marl_validation(
            wrapped_model, env, tracker, 1
        )
        
        # Check if we have enough data for score calculation
        if len(tracker.jobs_data) < 10 or len(tracker.carbon_window_data) < 10:
            if debug:
                print(f"  Insufficient data for score calculation (jobs: {len(tracker.jobs_data)}, windows: {len(tracker.carbon_window_data)})")
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
            f.write(f"Jobs analyzed: {len(scheduled_jobs)}\n")
            f.write(f"Carbon windows captured: {len(tracker.carbon_window_data)}\n\n")
            
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
                writer.writerow(['epoch', 'score', 'carbon_r2', 'wait_r2', 'carbon_coef', 'wait_coef', 'jobs_analyzed'])
            
            writer.writerow([
                epoch,
                score,
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
        
def train(workload, backfill, debug=False, experiment_name="", description=""):
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
        device          = device
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
            
        ppo.train()
        
        # --------------------------------------------------------------
        # 7. Logging: append averaged rewards to CSV
        # --------------------------------------------------------------
        avg_epoch_reward = epoch_reward / traj_num
        avg_green_reward = green_reward / traj_num
        avg_wait_reward = wait_reward / traj_num
        
        # Save to experiment directory
        csv_file = f"{experiment_dir}/training_results.csv"
        # Write header if file doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, mode="w", newline="") as file:
                csv.writer(file).writerow([
                    "epoch", "avg_epoch_reward", "avg_green_reward", "avg_wait_reward"
                ])
        
        with open(csv_file, mode="a", newline="") as file:
            csv.writer(file).writerow([
                epoch + 1,
                avg_epoch_reward,
                avg_green_reward,
                avg_wait_reward
            ])
        
        # Always show epoch summary (every epoch)
        print(f"Epoch {epoch + 1:3d}/{epochs} | Total: {avg_epoch_reward:7.3f} | Green: {avg_green_reward:7.3f} | Wait: {avg_wait_reward:7.3f}")
        
        # Save model weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{experiment_dir}/checkpoints/epoch_{epoch + 1}/"
            ppo.save_using_model_name(checkpoint_path)
            if debug:
                print(f"  → Checkpoint saved: epoch_{epoch + 1}/")
            
            # Calculate score every 5th epoch
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
    args = parser.parse_args()
    train(args.workload, args.backfill, args.debug, args.name, args.description)