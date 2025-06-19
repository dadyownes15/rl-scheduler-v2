#!/usr/bin/env python3
"""
Enhanced validation script for MARL experiments with comprehensive job tracking.
Fixes critical issues: job completion tracking, carbon emissions per job, and proper data logging.
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


def load_marl_model(experiment_path, workload_arg=None, epoch=None):
    """
    Load MARL model from experiment directory.
    
    Args:
        experiment_path: Experiment name (e.g., "MARL_basic", "basic") or full path
        workload_arg: Workload name or file path (e.g., "lublin_256_carbon_float" or "./data/lublin_256_carbon_float.swf")
        epoch: Epoch number to load weights from (if None, loads from final/)
    
    Returns:
        Loaded MARL model
    """
    inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
    featureNum_size = [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = MARL(batch_size=256, inputNum_size=inputNum_size,
                 featureNum_size=featureNum_size, device=device)
    
    # Auto-detect experiment path if only experiment name is provided
    if '/' not in experiment_path and workload_arg:
        # Extract workload name from argument
        if workload_arg.endswith('.swf'):
            # It's a file path like "./data/lublin_256_carbon_float.swf"
            workload_name = os.path.basename(workload_arg).replace('.swf', '')
        else:
            # It's just a workload name like "lublin_256_carbon_float"
            workload_name = workload_arg
        
        # Handle experiment name - check if it already has MARL_ prefix
        if experiment_path.startswith('MARL_'):
            # Already has prefix, use as-is
            full_experiment_path = f"{workload_name}/{experiment_path}"
        else:
            # Add MARL_ prefix
            full_experiment_path = f"{workload_name}/MARL_{experiment_path}"
        
        if os.path.exists(full_experiment_path):
            experiment_path = full_experiment_path
            print(f"Auto-detected experiment path: {experiment_path}")
        else:
            print(f"Warning: Could not auto-detect path for experiment '{experiment_path}', using as provided")
    
    # Determine weights path based on epoch
    if epoch is not None:
        # Load from specific epoch checkpoint
        weights_path = f"{experiment_path}/checkpoints/epoch_{epoch}/"
        if not os.path.exists(f"{weights_path}_actor.pkl"):
            # List available epochs to help user
            checkpoints_dir = f"{experiment_path}/checkpoints"
            if os.path.exists(checkpoints_dir):
                available_epochs = []
                for item in os.listdir(checkpoints_dir):
                    if item.startswith('epoch_') and os.path.isdir(f"{checkpoints_dir}/{item}"):
                        epoch_num = item.replace('epoch_', '')
                        if os.path.exists(f"{checkpoints_dir}/{item}/_actor.pkl"):
                            available_epochs.append(epoch_num)
                available_epochs.sort(key=int)
                print(f"Available epochs in {checkpoints_dir}/:")
                for epoch_num in available_epochs:
                    print(f"  - epoch_{epoch_num}")
            raise FileNotFoundError(f"No trained weights found for epoch {epoch} in {experiment_path}/checkpoints/")
    else:
        # Load from final weights, fallback to legacy location
        weights_path = f"{experiment_path}/final/"
        if not os.path.exists(f"{weights_path}_actor.pkl"):
            # Try legacy location
            workload = experiment_path.split('/')[0]
            weights_path = f"{workload}/MARL/"
            if not os.path.exists(f"{weights_path}_actor.pkl"):
                # List available experiments to help user
                if workload_arg:
                    # Extract workload name for listing
                    if workload_arg.endswith('.swf'):
                        workload_name = os.path.basename(workload_arg).replace('.swf', '')
                    else:
                        workload_name = workload_arg
                        
                    print(f"Available experiments in {workload_name}/:")
                    if os.path.exists(workload_name):
                        experiments = [d for d in os.listdir(workload_name) if d.startswith('MARL_')]
                        for exp in experiments:
                            print(f"  - {exp}")
                raise FileNotFoundError(f"No trained weights found in {experiment_path}")
    
    print(f"Loading weights from: {weights_path}")
    model.load_using_model_name(weights_path)
    
    # Store the final experiment path for later use
    model._experiment_path = experiment_path
    
    return model


class EnhancedJobTracker:
    """Enhanced job tracker with comprehensive completion and carbon tracking"""
    
    def __init__(self):
        self.jobs_data = []
        self.episode_data = []
        self.job_id_to_index = {}  # Map job_id to index for faster lookup
        self.carbon_window_data = []  # Store viewable window data for each job
        
    def record_job_submission(self, job, queue_length, current_time, env=None):
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
            'completed': False,
            'carbon_emissions': None,
            'carbon_reward': None
        }
        job_index = len(self.jobs_data)
        self.jobs_data.append(job_info)
        self.job_id_to_index[job.job_id] = job_index
        
        return job_index
    
    def record_agent_decision(self, env, selected_job, action1, action2):
        """
        Record the exact carbon window the agent saw when making this scheduling decision
        
        Args:
            env: Environment with carbon intensity data
            selected_job: Job that was selected for scheduling (or None)
            action1: Action 1 (job selection)
            action2: Action 2 (delay)
        """
        if selected_job is None:
            return
            
        try:
            # Get the EXACT carbon window that the agent saw when making this decision
            # This is the same call made in build_observation()
            carbon_slots = env.cluster.carbonIntensity.getCarbonIntensitySlot(env.current_timestamp)
            
            # Extract the raw intensities and times exactly as the agent sees them
            intensities = [slot['carbonIntensity'] for slot in carbon_slots]
            last_times = [slot['lastTime'] for slot in carbon_slots]
            
            # Calculate window statistics
            window_data = {
                'job_id': selected_job.job_id,
                'decision_timestamp': env.current_timestamp,
                'action1': action1.item() if hasattr(action1, 'item') else int(action1),
                'action2': action2.item() if hasattr(action2, 'item') else int(action2),
                # Raw carbon window data (exactly what agent sees)
                'carbon_intensities': intensities,
                'last_times': last_times,
                'window_start_time': env.current_timestamp,
                'window_duration_hours': len(intensities),
                # Statistical summaries
                'carbon_intensity_min': min(intensities) if intensities else 0,
                'carbon_intensity_max': max(intensities) if intensities else 0,
                'carbon_intensity_avg': sum(intensities) / len(intensities) if intensities else 0,
                'carbon_intensity_std': np.std(intensities) if intensities else 0,
                # Job characteristics
                'job_power': getattr(selected_job, 'power', 0),
                'job_runtime': selected_job.request_time,
                'job_processors': selected_job.request_number_of_processors,
                'job_carbon_consideration': getattr(selected_job, 'carbon_consideration', -1),
                # Queue state
                'queue_length_at_decision': len(env.job_queue),
                'running_jobs_count': len(env.running_jobs)
            }
            
            # Calculate immediate vs worst-case emissions using the same window
            if hasattr(env.cluster.carbonIntensity, 'getCarbonEmissions'):
                # Emissions if scheduled immediately
                immediate_emissions = env.cluster.carbonIntensity.getCarbonEmissions(
                    selected_job.power, env.current_timestamp, 
                    env.current_timestamp + selected_job.request_time
                )
                window_data['emissions_if_immediate'] = immediate_emissions
                
                # Worst-case emissions within this viewable window
                try:
                    window_end = env.current_timestamp + (len(intensities) * 3600)
                    worst_case_intensity = env.cluster.carbonIntensity.getMaxCarbonIntensityInPeriod(
                        env.current_timestamp, window_end, selected_job.request_time
                    )
                    energy_kwh = (selected_job.power / 1000.0) * (selected_job.request_time / 3600.0)
                    worst_case_emissions = energy_kwh * worst_case_intensity
                    window_data['worst_case_emissions'] = worst_case_emissions
                    window_data['worst_case_intensity'] = worst_case_intensity
                except Exception as e:
                    window_data['worst_case_emissions'] = None
                    window_data['worst_case_intensity'] = None
                    window_data['worst_case_error'] = str(e)
            
            self.carbon_window_data.append(window_data)
            
        except Exception as e:
            # Record minimal data if error occurs
            error_data = {
                'job_id': selected_job.job_id if selected_job else 'unknown',
                'decision_timestamp': env.current_timestamp,
                'action1': action1.item() if hasattr(action1, 'item') else int(action1),
                'action2': action2.item() if hasattr(action2, 'item') else int(action2),
                'error': str(e),
                'carbon_intensities': [],
                'last_times': [],
                'window_start_time': env.current_timestamp,
                'window_duration_hours': 0
            }
            self.carbon_window_data.append(error_data)
    
    def update_job_scheduling(self, job_index, job, scheduled_time, action1, action2):
        """Update job info when it gets scheduled with enhanced carbon calculation"""
        if job_index < len(self.jobs_data):
            # Convert tensor actions to plain numbers
            a1_val = action1.item() if hasattr(action1, 'item') else int(action1)
            a2_val = action2.item() if hasattr(action2, 'item') else int(action2)
            
            # Calculate individual job carbon emissions
            start_time = scheduled_time
            end_time = start_time + job.request_time
            
            # Create temporary single-job list for carbon calculation
            temp_job_list = [job]
            
            # Get carbon intensity object from environment (passed during initialization)
            if hasattr(self, 'carbon_intensity'):
                carbon_emissions = self.carbon_intensity.getCarbonEmissions(
                    job.power, start_time, end_time
                )
                
                # Calculate individual job carbon reward
                individual_carbon_reward = self.carbon_intensity.getCarbonAwareReward(temp_job_list)
            else:
                carbon_emissions = 0.0
                individual_carbon_reward = 0.0
            
            self.jobs_data[job_index].update({
                'scheduled_time': scheduled_time,
                'wait_time': scheduled_time - self.jobs_data[job_index]['submit_time'],
                'action1': a1_val,
                'action2': a2_val,
                'scheduled': True,
                'carbon_emissions': carbon_emissions,
                'carbon_reward': individual_carbon_reward
            })
    
    def update_job_completion(self, job_id, completion_time, actual_runtime):
        """Update job info when it completes - now uses job_id for direct lookup"""
        if job_id in self.job_id_to_index:
            job_index = self.job_id_to_index[job_id]
            self.jobs_data[job_index].update({
                'completion_time': completion_time,
                'actual_runtime': actual_runtime,
                'completed': True
            })
    
    def set_carbon_intensity(self, carbon_intensity):
        """Set carbon intensity object for calculations"""
        self.carbon_intensity = carbon_intensity
    
    def track_running_jobs_completion(self, env):
        """Track completion of currently running jobs based on their end times"""
        current_time = env.current_timestamp
        
        for job in env.running_jobs:
            job_end_time = job.scheduled_time + job.request_time
            if job_end_time <= current_time:
                # Job should be completed by now
                actual_runtime = job.request_time  # In simulation, actual = requested
                self.update_job_completion(job.job_id, job_end_time, actual_runtime)
    
    def finalize_remaining_jobs(self, env):
        """Mark remaining scheduled jobs as completed at episode end"""
        for job in env.loads.all_jobs:
            if (hasattr(job, 'scheduled_time') and job.scheduled_time != -1 and 
                job.job_id in self.job_id_to_index):
                
                job_data = self.jobs_data[self.job_id_to_index[job.job_id]]
                if not job_data['completed']:
                    completion_time = job.scheduled_time + job.request_time
                    actual_runtime = job.request_time
                    self.update_job_completion(job.job_id, completion_time, actual_runtime)
    
    def record_episode_summary(self, episode_num, total_reward, green_reward, jobs_completed, makespan):
        """Record episode-level summary"""
        completed_jobs = [j for j in self.jobs_data if j['completed']]
        avg_wait_time = np.mean([j['wait_time'] for j in completed_jobs if j['wait_time'] is not None])
        
        self.episode_data.append({
            'episode': episode_num,
            'total_reward': total_reward,
            'green_reward': green_reward,
            'jobs_completed': jobs_completed,
            'jobs_scheduled': len([j for j in self.jobs_data if j['scheduled']]),
            'jobs_actually_completed': len(completed_jobs),
            'makespan': makespan,
            'avg_wait_time': avg_wait_time
        })


def run_enhanced_marl_validation(model, env, tracker, episode_num):
    """
    Run a single validation episode with enhanced tracking.
    
    Args:
        model: Loaded MARL model
        env: HPC environment
        tracker: EnhancedJobTracker instance
        episode_num: Episode number for tracking
    
    Returns:
        (total_reward, green_reward, jobs_completed)
    """
    # Set carbon intensity for tracker
    tracker.set_carbon_intensity(env.cluster.carbonIntensity)
    
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
        
        # Track completion of running jobs before processing new ones
        tracker.track_running_jobs_completion(env)
        
        # Track new jobs in queue that we haven't seen before
        for job in env.job_queue:
            if job not in job_indices:
                job_idx = tracker.record_job_submission(
                    job, len(env.job_queue), env.current_timestamp, env
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
        
        # CRITICAL: Record the exact carbon window the agent saw when making this decision
        # This must be done BEFORE env.step() because the timestamp may change after
        tracker.record_agent_decision(env, selected_job, a1, a2)
        
        # Step environment
        o, r, d, r2, sjf_t, f1_t, running_num, greenRwd = env.step(a1, a2)
        
        # Update tracking for scheduled job with enhanced information
        if selected_job and selected_job in job_indices:
            job_idx = job_indices[selected_job]
            tracker.update_job_scheduling(
                job_idx, selected_job, env.current_timestamp, a1, a2
            )
        
        total_reward += r
        green_reward += greenRwd
        
        if d:
            jobs_completed = step_count
            break
        
        if step_count % 100 == 0:
            print(f"    Step {step_count}, queue size: {len(env.job_queue)}, running: {running_num}")
    
    # Finalize remaining jobs at episode end
    tracker.finalize_remaining_jobs(env)
    
    # Record episode summary
    makespan = env.current_timestamp - env.start_timestamp if hasattr(env, 'start_timestamp') else 0
    tracker.record_episode_summary(episode_num, total_reward, green_reward, jobs_completed, makespan)
    
    print(f"  Episode {episode_num} completed: steps={step_count}, reward={total_reward:.2f}, green={green_reward:.2f}")
    
    return total_reward, green_reward, jobs_completed


def save_enhanced_validation_results(tracker, experiment_path, workload, episodes, epoch=None):
    """Save enhanced validation results with comprehensive job completion data"""
    
    # Create validation results directory with epoch subfolder
    if epoch is not None:
        results_dir = f"{experiment_path}/validation_results/epoch_{epoch}"
    else:
        results_dir = f"{experiment_path}/validation_results/final"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save job-level data
    jobs_file = f"{results_dir}/job_details_enhanced.csv"
    if tracker.jobs_data:
        jobs_df = pd.DataFrame(tracker.jobs_data)
        jobs_df.to_csv(jobs_file, index=False)
        print(f"Enhanced job details saved to: {jobs_file}")
        
        # Print comprehensive statistics
        all_jobs = jobs_df
        scheduled_jobs = jobs_df[jobs_df['scheduled'] == True]
        completed_jobs = jobs_df[jobs_df['completed'] == True]
        
        print(f"\nEnhanced Job Statistics:")
        print(f"  Total jobs tracked: {len(all_jobs)}")
        print(f"  Jobs scheduled: {len(scheduled_jobs)}")
        print(f"  Jobs completed: {len(completed_jobs)}")
        
        if len(scheduled_jobs) > 0:
            print(f"  Average wait time: {scheduled_jobs['wait_time'].mean():.2f}s")
            
            # Carbon consideration analysis
            carbon_values = scheduled_jobs['carbon_consideration']
            print(f"  Carbon consideration stats:")
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
                print(f"    {bin_labels[i]}: {count} jobs ({count/len(scheduled_jobs)*100:.1f}%)")
        
        if len(completed_jobs) > 0:
            # Carbon emissions analysis
            emissions_data = completed_jobs['carbon_emissions'].dropna()
            if len(emissions_data) > 0:
                print(f"  Carbon emissions stats:")
                print(f"    Total emissions: {emissions_data.sum():.2f} gCO2eq")
                print(f"    Mean per job: {emissions_data.mean():.2f} gCO2eq")
                print(f"    Std: {emissions_data.std():.2f} gCO2eq")
            
            # Carbon reward analysis
            reward_data = completed_jobs['carbon_reward'].dropna()
            reward_data = reward_data[reward_data != 0]  # Exclude zero rewards
            if len(reward_data) > 0:
                print(f"  Carbon reward stats (non-zero):")
                print(f"    Mean: {reward_data.mean():.4f}")
                print(f"    Std: {reward_data.std():.4f}")
                print(f"    Min: {reward_data.min():.4f}")
                print(f"    Max: {reward_data.max():.4f}")
            
            # Completion analysis
            print(f"  Completion analysis:")
            print(f"    Jobs with completion data: {len(completed_jobs)}")
            print(f"    Completion rate: {len(completed_jobs)/len(scheduled_jobs)*100:.1f}%")

    # Save episode-level data
    episodes_file = f"{results_dir}/episode_summary_enhanced.csv"
    if tracker.episode_data:
        episodes_df = pd.DataFrame(tracker.episode_data)
        episodes_df.to_csv(episodes_file, index=False)
        print(f"Enhanced episode summary saved to: {episodes_file}")
    
    # Save carbon window data
    carbon_window_file = f"{results_dir}/carbon_window_data.csv"
    if tracker.carbon_window_data:
        # Flatten the carbon window data for CSV storage
        flattened_window_data = []
        for window in tracker.carbon_window_data:
            base_data = {
                'job_id': window.get('job_id', 'unknown'),
                'decision_timestamp': window.get('decision_timestamp', 0),
                'action1': window.get('action1', -1),
                'action2': window.get('action2', -1),
                'window_start_time': window.get('window_start_time', 0),
                'window_duration_hours': window.get('window_duration_hours', 0),
                'carbon_intensity_min': window.get('carbon_intensity_min', 0),
                'carbon_intensity_max': window.get('carbon_intensity_max', 0),
                'carbon_intensity_avg': window.get('carbon_intensity_avg', 0),
                'carbon_intensity_std': window.get('carbon_intensity_std', 0),
                'job_power': window.get('job_power', 0),
                'job_runtime': window.get('job_runtime', 0),
                'job_processors': window.get('job_processors', 0),
                'job_carbon_consideration': window.get('job_carbon_consideration', -1),
                'queue_length_at_decision': window.get('queue_length_at_decision', 0),
                'running_jobs_count': window.get('running_jobs_count', 0),
                'emissions_if_immediate': window.get('emissions_if_immediate', None),
                'worst_case_emissions': window.get('worst_case_emissions', None),
                'worst_case_intensity': window.get('worst_case_intensity', None),
                'error': window.get('error', None)
            }
            
            # Add individual carbon intensities as separate columns
            intensities = window.get('carbon_intensities', [])
            for i, intensity in enumerate(intensities):
                base_data[f'carbon_intensity_hour_{i}'] = intensity
            
            # Add last times as separate columns
            last_times = window.get('last_times', [])
            for i, last_time in enumerate(last_times):
                base_data[f'last_time_hour_{i}'] = last_time
            
            flattened_window_data.append(base_data)
        
        window_df = pd.DataFrame(flattened_window_data)
        window_df.to_csv(carbon_window_file, index=False)
        print(f"Carbon window data saved to: {carbon_window_file}")
        
        # Print window data statistics
        if len(flattened_window_data) > 0:
            print(f"\nCarbon Window Statistics:")
            print(f"  Windows captured: {len(flattened_window_data)}")
            print(f"  Average window duration: {window_df['window_duration_hours'].mean():.1f} hours")
            print(f"  Carbon intensity range:")
            print(f"    Min across all windows: {window_df['carbon_intensity_min'].min():.2f} gCO2eq/kWh")
            print(f"    Max across all windows: {window_df['carbon_intensity_max'].max():.2f} gCO2eq/kWh")
            print(f"    Average intensity: {window_df['carbon_intensity_avg'].mean():.2f} gCO2eq/kWh")


def main():
    """Enhanced main function with improved job tracking"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced MARL Validation with Comprehensive Job Tracking')
    parser.add_argument('--experiment', required=True, help='Experiment name (e.g., MARL_basic, basic) or full path')
    parser.add_argument('--workload', required=True, help='Workload name (e.g., lublin_256_carbon_float) or file path (e.g., ./data/lublin_256_carbon_float.swf)')
    parser.add_argument('--epoch', type=int, help='Epoch number to load weights from (if not specified, loads from final/)')
    parser.add_argument('--episodes', type=int, default=5, help='Number of validation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--backfill', type=int, default=0, help='Backfill strategy (0=FCFS, 1=backfill enabled)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENHANCED MARL VALIDATION WITH COMPREHENSIVE JOB TRACKING")
    print("=" * 80)
    print(f"Experiment: {args.experiment}")
    print(f"Workload: {args.workload}")
    print(f"Epoch: {args.epoch if args.epoch is not None else 'final'}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print(f"Backfill: {args.backfill}")
    print(f"Debug: {args.debug}")
    print()
    
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model
    print("Loading MARL model...")
    model = load_marl_model(args.experiment, args.workload, args.epoch)
    print("✓ Model loaded successfully")
    
    # Initialize environment
    print("Initializing environment...")
    
    # Determine workload file path
    if args.workload.endswith('.swf'):
        # It's already a file path
        workload_file = args.workload
    else:
        # It's just a workload name, construct file path
        workload_file = f"./data/{args.workload}.swf"
        print(f"Using workload file: {workload_file}")
    
    env = HPCEnv(backfill=args.backfill, debug=args.debug)
    env.my_init(workload_file)
    env.seed(args.seed)
    print("✓ Environment initialized")
    
    # Initialize enhanced tracker
    tracker = EnhancedJobTracker()
    
    # Run validation episodes
    print(f"\nRunning {args.episodes} validation episodes...")
    episode_results = []
    
    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        env.reset()
        
        total_reward, green_reward, jobs_completed = run_enhanced_marl_validation(
            model, env, tracker, episode + 1
        )
        
        episode_results.append({
            'episode': episode + 1,
            'total_reward': total_reward,
            'green_reward': green_reward,
            'jobs_completed': jobs_completed
        })
    
    # Save results
    print(f"\nSaving enhanced validation results...")
    # Pass the detected experiment path from model loading
    if hasattr(model, '_experiment_path'):
        experiment_path = model._experiment_path
    else:
        experiment_path = args.experiment
    save_enhanced_validation_results(tracker, experiment_path, workload_file, args.episodes, args.epoch)
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("ENHANCED VALIDATION SUMMARY")
    print("=" * 80)
    
    total_rewards = [r['total_reward'] for r in episode_results]
    green_rewards = [r['green_reward'] for r in episode_results]
    
    print(f"Total Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Green Reward: {np.mean(green_rewards):.4f} ± {np.std(green_rewards):.4f}")
    print(f"Episodes completed: {len(episode_results)}")
    
    print(f"\n✅ Enhanced validation completed successfully!")
    if args.epoch is not None:
        print(f"Check {args.experiment}/validation_results/epoch_{args.epoch}/ for detailed results")
    else:
        print(f"Check {args.experiment}/validation_results/final/ for detailed results")


if __name__ == "__main__":
    main() 