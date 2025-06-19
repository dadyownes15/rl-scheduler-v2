#!/usr/bin/env python3
"""
Validation script for MARL experiments.
Loads trained weights and runs detailed job-level evaluation.
"""

import pandas as pd
import os
import random
import torch
import csv
import numpy as np
from datetime import datetime
from HPCSimPickJobs import *
from MARL import PPO as MARL


def load_marl_model(experiment_path):
    """
    Load MARL model from experiment directory.
    
    Args:
        experiment_path: Path to experiment directory (e.g., "lublin_256/MARL_ED12")
    
    Returns:
        Loaded MARL model
    """
    inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
    featureNum_size = [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = MARL(batch_size=256, inputNum_size=inputNum_size,
                 featureNum_size=featureNum_size, device=device)
    
    # Try to load from final weights, fallback to legacy location
    weights_path = f"{experiment_path}/final/"
    if not os.path.exists(f"{weights_path}_actor.pkl"):
        # Try legacy location
        workload = experiment_path.split('/')[0]
        weights_path = f"{workload}/MARL/"
        if not os.path.exists(f"{weights_path}_actor.pkl"):
            raise FileNotFoundError(f"No trained weights found in {experiment_path}")
    
    print(f"Loading weights from: {weights_path}")
    model.load_using_model_name(weights_path)
    return model


class JobTracker:
    """Track detailed job-level information during validation"""
    
    def __init__(self):
        self.jobs_data = []
        self.episode_data = []
        
    def record_job_submission(self, job, queue_length, current_time):
        """Record job when it's submitted to the queue"""
        job_info = {
            'job_id': getattr(job, 'job_id', 'unknown'),
            'submit_time': job.submit_time,
            'request_time': job.request_time,
            'request_processors': job.request_number_of_processors,
            'carbon_consideration': getattr(job, 'carbon_consideration', -1),
            'queue_length_at_submission': queue_length,
            'submission_timestamp': current_time,
            'user_id': getattr(job, 'user_id', -1),
            'power': getattr(job, 'power', 0),
            # Will be filled when scheduled/completed
            'scheduled_time': None,
            'completion_time': None,
            'wait_time': None,
            'actual_runtime': None,
            'action1': None,
            'action2': None,
            'scheduled': False,
            'carbon_emissions': None,
            'carbon_reward': None
        }
        self.jobs_data.append(job_info)
        return len(self.jobs_data) - 1  # Return index for later updates
    
    def update_job_scheduling(self, job_index, scheduled_time, action1, action2, carbon_emissions=None, carbon_reward=None):
        """Update job info when it gets scheduled"""
        if job_index < len(self.jobs_data):
            self.jobs_data[job_index].update({
                'scheduled_time': scheduled_time,
                'wait_time': scheduled_time - self.jobs_data[job_index]['submit_time'],
                'action1': action1,
                'action2': action2,
                'scheduled': True,
                'carbon_emissions': carbon_emissions,
                'carbon_reward': carbon_reward
            })
    
    def update_job_completion(self, job_index, completion_time, actual_runtime):
        """Update job info when it completes"""
        if job_index < len(self.jobs_data):
            self.jobs_data[job_index].update({
                'completion_time': completion_time,
                'actual_runtime': actual_runtime
            })
    
    def record_episode_summary(self, episode_num, total_reward, green_reward, jobs_completed, makespan):
        """Record episode-level summary"""
        self.episode_data.append({
            'episode': episode_num,
            'total_reward': total_reward,
            'green_reward': green_reward,
            'jobs_completed': jobs_completed,
            'makespan': makespan,
            'avg_wait_time': np.mean([j['wait_time'] for j in self.jobs_data if j['wait_time'] is not None])
        })


def run_marl_validation(model, env, tracker, episode_num):
    """
    Run a single validation episode with detailed tracking.
    
    Args:
        model: Loaded MARL model
        env: HPC environment
        tracker: JobTracker instance
        episode_num: Episode number for tracking
    
    Returns:
        (total_reward, green_reward, jobs_completed)
    """
    o = env.build_observation()
    running_num = 0
    total_reward = 0
    green_reward = 0
    jobs_completed = 0
    job_indices = {}  # Map job objects to tracker indices
    
    print(f"  Starting episode {episode_num}, initial queue size: {len(env.job_queue)}")
    
    step_count = 0
    while True:
        step_count += 1
        
        # Track new jobs in queue that we haven't seen before
        for job in env.job_queue:
            if job not in job_indices:
                job_idx = tracker.record_job_submission(
                    job, len(env.job_queue), env.current_timestamp
                )
                job_indices[job] = job_idx
        
        # Build action mask (same logic as compare.py)
        lst = []
        for i in range(0, MAX_QUEUE_SIZE * JOB_FEATURES, JOB_FEATURES):
            job_slice = o[i:i + JOB_FEATURES]
            # Check for padding patterns
            padding_pattern1 = [0] + [1] * (JOB_FEATURES - 2) + [0]
            padding_pattern2 = [1] * JOB_FEATURES
            
            if (len(job_slice) == len(padding_pattern1) and 
                all(abs(job_slice[j] - padding_pattern1[j]) < 1e-6 for j in range(len(job_slice)))):
                lst.append(1)  # Mask out (invalid)
            elif (len(job_slice) == len(padding_pattern2) and 
                  all(abs(job_slice[j] - padding_pattern2[j]) < 1e-6 for j in range(len(job_slice)))):
                lst.append(1)  # Mask out (invalid)
            else:
                lst.append(0)  # Valid job
        
        # Action 2 mask
        mask2 = np.zeros(action2_num, dtype=int)
        if running_num < delayMaxJobNum:
            mask2[running_num + 1:delayMaxJobNum + 1] = 1
        
        # Get actions from model
        a1, a2 = model.eval_action(o, lst, mask2)
        
        # Record which job was selected (if any)
        selected_job = None
        if a1 < len(env.job_queue):
            selected_job = env.job_queue[a1]
        
        # Step environment
        o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = env.step(a1, a2)
        
        # Update tracking for scheduled job
        if selected_job and selected_job in job_indices:
            job_idx = job_indices[selected_job]
            tracker.update_job_scheduling(
                job_idx, env.current_timestamp, a1, a2, 
                carbon_emissions=None,  # Could extract from env if available
                carbon_reward=greenRwd
            )
        
        total_reward += r
        green_reward += greenRwd
        
        if d:
            jobs_completed = step_count
            break
        
        if step_count % 100 == 0:
            print(f"    Step {step_count}, queue size: {len(env.job_queue)}, running: {running_num}")
    
    # Record episode summary
    makespan = env.current_timestamp - env.start_timestamp if hasattr(env, 'start_timestamp') else 0
    tracker.record_episode_summary(episode_num, total_reward, green_reward, jobs_completed, makespan)
    
    print(f"  Episode {episode_num} completed: steps={step_count}, reward={total_reward:.2f}, green={green_reward:.2f}")
    
    return total_reward, green_reward, jobs_completed


def save_validation_results(tracker, experiment_path, workload, episodes):
    """Save detailed validation results to CSV files"""
    
    # Create validation results directory
    results_dir = f"{experiment_path}/validation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save job-level data
    jobs_file = f"{results_dir}/job_details_{timestamp}.csv"
    if tracker.jobs_data:
        jobs_df = pd.DataFrame(tracker.jobs_data)
        jobs_df.to_csv(jobs_file, index=False)
        print(f"Job details saved to: {jobs_file}")
        
        # Print summary statistics
        completed_jobs = jobs_df[jobs_df['scheduled'] == True]
        if len(completed_jobs) > 0:
            print(f"\nJob Statistics:")
            print(f"  Total jobs tracked: {len(jobs_df)}")
            print(f"  Jobs scheduled: {len(completed_jobs)}")
            print(f"  Average wait time: {completed_jobs['wait_time'].mean():.2f}s")
                         print(f"  Carbon consideration distribution:")
             carbon_values = completed_jobs['carbon_consideration']
             print(f"    Min: {carbon_values.min():.3f}")
             print(f"    Max: {carbon_values.max():.3f}")
             print(f"    Mean: {carbon_values.mean():.3f}")
             print(f"    Std: {carbon_values.std():.3f}")
             
             # Show distribution in bins
             bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
             bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
             for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                 count = len(carbon_values[(carbon_values >= low) & (carbon_values < high)])
                 if i == len(bins) - 2:  # Last bin, include upper bound
                     count = len(carbon_values[(carbon_values >= low) & (carbon_values <= high)])
                 print(f"    {bin_labels[i]}: {count} jobs ({count/len(completed_jobs)*100:.1f}%)")
    
    # Save episode-level data
    episodes_file = f"{results_dir}/episode_summary_{timestamp}.csv"
    if tracker.episode_data:
        episodes_df = pd.DataFrame(tracker.episode_data)
        episodes_df.to_csv(episodes_file, index=False)
        print(f"Episode summary saved to: {episodes_file}")
        
        # Print episode statistics
        print(f"\nEpisode Statistics:")
        print(f"  Episodes completed: {len(episodes_df)}")
        print(f"  Average total reward: {episodes_df['total_reward'].mean():.2f}")
        print(f"  Average green reward: {episodes_df['green_reward'].mean():.2f}")
        print(f"  Average jobs per episode: {episodes_df['jobs_completed'].mean():.1f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate MARL experiment with detailed job tracking")
    parser.add_argument('--workload', type=str, default='lublin_256', 
                       help='Workload dataset name')
    parser.add_argument('--experiment', type=str, required=True,
                       help='Experiment name (e.g., ED12)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of validation episodes to run')
    parser.add_argument('--backfill', type=int, default=0,
                       help='Backfill strategy (0/1)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    parser.add_argument('--sequence_length', type=int, default=1024,
                       help='Length of job sequence to evaluate')
    
    args = parser.parse_args()
    
    # Setup paths
    experiment_path = f"{args.workload}/MARL_{args.experiment}"
    workload_file = f"data/{args.workload}.swf"
    
    print(f"MARL Validation")
    print(f"===============")
    print(f"Workload: {args.workload}")
    print(f"Experiment: {args.experiment}")
    print(f"Episodes: {args.episodes}")
    print(f"Backfill: {args.backfill}")
    print(f"Experiment path: {experiment_path}")
    
    # Check if experiment exists
    if not os.path.exists(experiment_path):
        print(f"ERROR: Experiment directory not found: {experiment_path}")
        return 1
    
    # Check if workload file exists
    if not os.path.exists(workload_file):
        print(f"ERROR: Workload file not found: {workload_file}")
        return 1
    
    try:
        # Load model
        print(f"\nLoading MARL model...")
        model = load_marl_model(experiment_path)
        print(f"✓ Model loaded successfully")
        
        # Initialize environment
        print(f"Initializing environment...")
        env = HPCEnv(backfill=args.backfill, debug=args.debug)
        env.my_init(workload_file=workload_file)
        print(f"✓ Environment initialized")
        
        # Initialize tracker
        tracker = JobTracker()
        
        # Run validation episodes
        print(f"\nRunning {args.episodes} validation episodes...")
        
        # Use fixed seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        start_positions = [random.randint(100, env.loads.size() - args.sequence_length - 100) 
                          for _ in range(args.episodes)]
        
        total_rewards = []
        green_rewards = []
        
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            
            # Reset environment for this episode
            env.reset_for_test(args.sequence_length, start_positions[episode])
            
            # Run episode
            total_reward, green_reward, jobs_completed = run_marl_validation(
                model, env, tracker, episode + 1
            )
            
            total_rewards.append(total_reward)
            green_rewards.append(green_reward)
        
        # Print overall results
        print(f"\n" + "="*50)
        print(f"VALIDATION COMPLETE")
        print(f"="*50)
        print(f"Average total reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average green reward: {np.mean(green_rewards):.2f} ± {np.std(green_rewards):.2f}")
        
        # Save results
        save_validation_results(tracker, experiment_path, args.workload, args.episodes)
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 