import os

import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.signal
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
        
def train(workload, backfill, debug=False):
    # ------------------------------------------------------------------
    # 1. Experiment-wide hyper-parameters & environment construction
    # ------------------------------------------------------------------
    seed       = 0
    epochs     = 300          # how many training loops over the dataset
    traj_num   = 100          # trajectories per epoch (episodes)
    
    if debug:
        print(f"DEBUG: Training parameters:")
        print(f"  Workload: {workload}")
        print(f"  Backfill: {backfill}")
        print(f"  Epochs: {epochs}")
        print(f"  Trajectories per epoch: {traj_num}")
        print(f"  Seed: {seed}")
    
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
        while True:
            if debug and t < 3:  # Only debug first few trajectories
                print(f"  DEBUG: Trajectory {t + 1}, step {ep_len}, running jobs: {running_num}")
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
            
            # Debug mask creation
            if debug and t < 3 and ep_len < 5:
                valid_jobs = lst.count(0)
                print(f"    MASK DEBUG: Valid jobs found: {valid_jobs}/{MAX_QUEUE_SIZE}")
                print(f"    MASK DEBUG: First 10 mask values: {lst[:10]}")
                print(f"    MASK DEBUG: Job queue size from env: {len(env.job_queue)}")
                # Check the expected padding pattern for 7 features
                expected_padding = [0, 1, 1, 1, 1, 0.5, 0]
                print(f"    MASK DEBUG: Expected padding pattern (7 features): {expected_padding}")
                # Check a few job slices
                for idx in range(min(3, MAX_QUEUE_SIZE)):
                    start = idx * JOB_FEATURES
                    end = start + JOB_FEATURES
                    slice_data = o[start:end]
                    mask_val = lst[idx]
                    print(f"    MASK DEBUG: Job {idx}: mask={mask_val}, features={slice_data}")
            
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
            
            # ----------------------------------------------------------
            # 5-c. Step the environment
            # ----------------------------------------------------------
            if debug and t < 3 and ep_len < 5:  # Only debug first few steps of first trajectories
                print(f"    Action1: {action1.item()}, Action2: {action2.item()}")
                
            o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = \
                env.step(action1.item(), action2.item())
                
            if debug and t < 3 and ep_len < 5:
                print(f"    Reward: {r:.4f}, Green reward: {greenRwd:.4f}, Done: {d}")
            
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
                if debug and t < 3:
                    print(f"  DEBUG: Episode {t + 1} completed, episode return: {ep_ret:.4f}, steps: {ep_len}")
                    
                t += 1                                     # finished a trajectory
                ppo.storeIntoBuffter(eta * r + greenRwd)   # final R for GAE
                ppo.clear_memory()                         # reset per-traj state
                
                # reset env for next trajectory
                o, r, d, ep_ret, ep_len, show_ret, sjf, f1, greenRwd = (
                    env.reset(), 0, False, 0, 0, 0, 0, 0, 0
                )
                running_num = 0
                
                if t >= traj_num:     # collected enough rollouts for this epoch
                    if debug:
                        print(f"  DEBUG: Epoch {epoch + 1} data collection complete ({t} trajectories)")
                    break
        
        # --------------------------------------------------------------
        # 6. Policy / value-function update after traj_num rollouts
        # --------------------------------------------------------------
        if debug:
            print(f"  DEBUG: Starting policy update for epoch {epoch + 1}")
            
        ppo.train()
        
        if debug:
            print(f"  DEBUG: Policy update complete")
        
        # --------------------------------------------------------------
        # 7. Logging: append averaged rewards to CSV
        # --------------------------------------------------------------
        avg_epoch_reward = epoch_reward / traj_num
        avg_green_reward = green_reward / traj_num
        avg_wait_reward = wait_reward / traj_num
        
        with open(f"MARL_{workload}.csv", mode="a", newline="") as file:
            csv.writer(file).writerow([
                avg_epoch_reward,
                avg_green_reward,
                avg_wait_reward
            ])
        
        if debug:
            print(f"  DEBUG: Epoch {epoch + 1} summary:")
            print(f"    Total reward: {avg_epoch_reward:.4f}")
            print(f"    Green reward: {avg_green_reward:.4f}")
            print(f"    Wait reward: {avg_wait_reward:.4f}")
        elif (epoch + 1) % 10 == 0:  # Print every 10 epochs when not in debug mode
            print(f"Epoch {epoch + 1}/{epochs} - Total: {avg_epoch_reward:.4f}, Green: {avg_green_reward:.4f}, Wait: {avg_wait_reward:.4f}")
        
        # Save model weights every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{workload}/MARL_checkpoints/epoch_{epoch + 1}/"
            ppo.save_using_model_name(checkpoint_path)
            if debug or (epoch + 1) % 10 == 0:
                print(f"  Saved checkpoint at epoch {epoch + 1}: {checkpoint_path}")
        
        # clear buffers so next epoch starts fresh
        ppo.buffer.clear_buffer()
    
    # ------------------------------------------------------------------
    # 8. Persist the trained model
    # ------------------------------------------------------------------
    ppo.save_using_model_name(f"{workload}/MARL/")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='lublin_256')
    parser.add_argument('--backfill', type=int, default=0)
    parser.add_argument('--debug', action='store_true', help='Enable debug prints')
    args = parser.parse_args()
    train(args.workload, args.backfill, args.debug)