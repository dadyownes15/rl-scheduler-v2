#!/usr/bin/env python3
"""
Interactive Episode Scheduling Visualizer

This program loads a trained actor model and visualizes the scheduling decisions
in a real-time interface with interactive controls.

Features:
- Real-time timeline visualization of scheduled jobs
- Live carbon intensity graph
- Job queue display with priorities
- Interactive step-through controls with action history
- Statistics and metrics display

Usage:
    python visualize_episode_scheduling.py <model_path> [options]

Examples:
    python visualize_episode_scheduling.py lublin_256_carbon_float_simple/MARL_curriculum_backfill_fcfs_v4 --epoch 55
    python visualize_episode_scheduling.py lublin_256_carbon_float_simple/MARL_curriculum_backfill_fcfs_v4
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import configparser
import copy

# Add current directory to path
sys.path.append(os.getcwd())

from HPCSimPickJobs import HPCEnv
from MARL_backfill import PPO, PPOModelWrapper

# Set up matplotlib for interactive plotting
plt.ion()
sns.set_style("whitegrid")

class LegacyModelWrapper:
    """Wrapper for older models that don't have action3"""
    def __init__(self, old_ppo):
        self.old_ppo = old_ppo
    
    def eval_action(self, observation, mask1, mask2, mask3):
        """Evaluate action using legacy model interface"""
        # Ensure observation is numpy array for reshape operation
        if isinstance(observation, list):
            observation = np.array(observation)
        # Ensure masks are numpy arrays
        if isinstance(mask1, list):
            mask1 = np.array(mask1)
        if isinstance(mask2, list):
            mask2 = np.array(mask2)
        
        # Legacy models only return action1 and action2
        action1, action2 = self.old_ppo.eval_action(observation, mask1, mask2)
        # For action3, just return 0 as placeholder
        action3 = torch.tensor([0])
        return action1, action2, action3

class FixedPPOModelWrapper:
    """Wrapper for newer models that handles list to numpy conversion"""
    def __init__(self, ppo_model):
        self.ppo_model = ppo_model
        
    def eval_action(self, observation, mask1, mask2, mask3):
        """Wrapper for eval_action with data type fixing"""
        # Ensure observation is numpy array for reshape operation
        if isinstance(observation, list):
            observation = np.array(observation)
        # Ensure masks are numpy arrays
        if isinstance(mask1, list):
            mask1 = np.array(mask1)
        if isinstance(mask2, list):
            mask2 = np.array(mask2)
        if isinstance(mask3, list):
            mask3 = np.array(mask3)
        
        return self.ppo_model.eval_action(observation, mask1, mask2, mask3)

class SchedulingVisualizer:
    def __init__(self, actor_weights_path, workload="lublin_256_carbon", backfill=3, debug=False):
        """Initialize the scheduling visualizer"""
        self.actor_weights_path = actor_weights_path
        self.workload = workload
        self.backfill = backfill
        self.debug = debug
        
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read('configFile/config.ini')
        
        # Initialize environment
        self.env = HPCEnv(backfill=backfill, debug=False)
        self.env.seed(42)  # Fixed seed for reproducible visualization
        
        current_dir = os.getcwd()
        workload_file = os.path.join(current_dir, "data", f"{workload}.swf")
        
        if not os.path.exists(workload_file):
            raise FileNotFoundError(f"Workload file not found: {workload_file}")
        
        self.env.my_init(workload_file=workload_file)
        
        # Load model
        self.model = self._load_model()
        
        # Initialize plots
        self._setup_plots()
        
        # Episode state management
        self.episode_seed = 42
        self.action_history = []  # Store actions for reset-and-replay back navigation
        self.current_step = 0
        self.done = False
        
        # Current environment state
        self.observation = None
        self.current_time = 0
        
        # Color schemes
        self.job_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.priority_colors = {
            'high': '#FF6B6B',     # Red - High carbon consideration
            'medium': '#4ECDC4',   # Teal - Medium carbon consideration
            'low': '#45B7D1',      # Blue - Low carbon consideration
            'backfill': '#96CEB4', # Green
            'running': '#FECA57'   # Yellow
        }
    
    def _load_model(self):
        """Load the trained actor model"""
        print(f"Loading model from: {self.actor_weights_path}")
        
        # Get model dimensions from config
        MAX_QUEUE_SIZE = int(self.config.get('GAS-MARL setting', 'max_queue_size'))
        run_win = int(self.config.get('GAS-MARL setting', 'run_win'))
        green_win = int(self.config.get('GAS-MARL setting', 'green_win'))
        JOB_FEATURES = 7
        RUN_FEATURE = 7
        GREEN_FEATURE = 7
        
        inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
        
        # Check if actor file exists
        actor_file = os.path.join(self.actor_weights_path, "_actor.pkl")
        if not os.path.exists(actor_file):
            raise FileNotFoundError(f"Actor weights not found: {actor_file}")
        
        # Try loading with neural backfill - try different feature configurations
        feature_configs = [
            [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE],  # Current format: [7, 7, 7]
            [JOB_FEATURES, 4, 2],  # Old format: [7, 4, 2]
        ]
        
        for features in feature_configs:
            try:
                print(f"Trying neural backfill model with feature config: {features}")
                ppo = PPO(
                    batch_size=256,
                    inputNum_size=inputNum_size,
                    featureNum_size=features,
                    device='cpu',
                    debug=False,
                    backfill=self.backfill
                )
                
                ppo.load_using_model_name(self.actor_weights_path + "/")
                print(f"✅ Model loaded successfully (neural backfill, features: {features})")
                return FixedPPOModelWrapper(ppo)
                    
            except Exception as e:
                print(f"  Failed with config {features}: {e}")
                continue
        
        print("🔄 Trying with older model architecture...")
        
        # Try with older MARL architecture (no backfill parameter)
        try:
            from MARL import PPO as OldPPO
            
            # For older models, try different feature sizes
            old_feature_configs = [
                [JOB_FEATURES, 4, 2],  # Old format
                [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE],  # Current format
            ]
            
            for features in old_feature_configs:
                try:
                    print(f"  Trying legacy model with feature config: {features}")
                    old_ppo = OldPPO(
                        batch_size=256,
                        inputNum_size=inputNum_size,
                        featureNum_size=features,
                        device='cpu',
                        debug=False
                    )
                    
                    old_ppo.load_using_model_name(self.actor_weights_path + "/")
                    print("✅ Model loaded successfully (legacy format)")
                    
                    return LegacyModelWrapper(old_ppo)
                    
                except Exception as inner_e:
                    print(f"    Failed with config {features}: {inner_e}")
                    continue
                    
        except ImportError:
            print("❌ Could not import legacy MARL module")
        
        raise FileNotFoundError(f"Could not load model from: {actor_file}")

    def _setup_plots(self):
        """Setup the interactive plot interface"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(24, 14))
        self.fig.suptitle('Interactive Episode Scheduling Visualizer', fontsize=16, fontweight='bold')
        
        # Create grid layout (5x4 to accommodate finished jobs panel)
        gs = self.fig.add_gridspec(5, 4, hspace=0.3, wspace=0.3)
        
        # Timeline view (top, spans 3 columns)
        self.ax_timeline = self.fig.add_subplot(gs[0:2, 0:3])
        self.ax_timeline.set_title('Job Scheduling Timeline', fontweight='bold')
        self.ax_timeline.set_xlabel('Time (hours)')
        self.ax_timeline.set_ylabel('Processor Slots')
        
        # Carbon intensity graph (top right)
        self.ax_carbon = self.fig.add_subplot(gs[0, 3])
        self.ax_carbon.set_title('Carbon Intensity', fontweight='bold')
        self.ax_carbon.set_xlabel('Time (hours)')
        self.ax_carbon.set_ylabel('gCO2eq/kWh')
        
        # Job queue view (middle right)
        self.ax_queue = self.fig.add_subplot(gs[1, 3])
        self.ax_queue.set_title('Job Queue', fontweight='bold')
        self.ax_queue.set_xlim(0, 10)
        self.ax_queue.set_ylim(0, 10)
        self.ax_queue.axis('off')
        
        # Finished jobs panel (new)
        self.ax_finished = self.fig.add_subplot(gs[2, 3])
        self.ax_finished.set_title('Recently Finished Jobs', fontweight='bold')
        self.ax_finished.set_xlim(0, 10)
        self.ax_finished.set_ylim(0, 10)
        self.ax_finished.axis('off')
        
        # Statistics panel (bottom left)
        self.ax_stats = self.fig.add_subplot(gs[2, 0:2])
        self.ax_stats.set_title('Episode Statistics', fontweight='bold')
        self.ax_stats.axis('off')
        
        # Action information (bottom middle)
        self.ax_action = self.fig.add_subplot(gs[2, 2])
        self.ax_action.set_title('Current Action', fontweight='bold')
        self.ax_action.axis('off')
        
        # Controls (moved down one row)
        self.ax_controls = self.fig.add_subplot(gs[3, 3])
        self.ax_controls.set_title('Controls', fontweight='bold')
        self.ax_controls.axis('off')
        
        # Running jobs display (spans bottom rows)
        self.ax_running = self.fig.add_subplot(gs[3:5, :])
        self.ax_running.set_title('Currently Running Jobs', fontweight='bold')
        self.ax_running.set_xlim(0, 10)
        self.ax_running.set_ylim(0, 5)
        self.ax_running.axis('off')
        
        # Add control buttons
        self._setup_controls()
        
        # Apply layout with warning suppression
        try:
            plt.tight_layout()
        except UserWarning:
            pass
    
    def _setup_controls(self):
        """Setup interactive controls"""
        # Step button
        ax_step = plt.axes([0.85, 0.02, 0.05, 0.03])
        self.btn_step = Button(ax_step, 'Step')
        self.btn_step.on_clicked(self.step_episode)
        
        # Back button
        ax_back = plt.axes([0.79, 0.02, 0.05, 0.03])
        self.btn_back = Button(ax_back, 'Back')
        self.btn_back.on_clicked(self.step_back)
        
        # Reset button  
        ax_reset = plt.axes([0.73, 0.02, 0.05, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_episode)
        
        # Auto-play button
        ax_auto = plt.axes([0.67, 0.02, 0.05, 0.03])
        self.btn_auto = Button(ax_auto, 'Auto')
        self.btn_auto.on_clicked(self.toggle_auto_play)
        
        # Speed slider
        ax_speed = plt.axes([0.3, 0.02, 0.3, 0.025])
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0)
        
        self.auto_play = False
        self.play_speed = 1.0
        self.auto_play_timer = None

    def reset_episode(self, event=None):
        """Reset episode to beginning"""
        print("\n🔄 RESET EPISODE REQUESTED")
        self._print_visual_state("BEFORE RESET")
        
        # Stop auto-play if running
        if hasattr(self, 'auto_play') and self.auto_play:
            self.auto_play = False
            self.btn_auto.label.set_text('Auto')
            if hasattr(self, 'auto_play_timer') and self.auto_play_timer:
                try:
                    self.auto_play_timer.stop()
                except:
                    pass
        
        print(f"  - Clearing {len(self.action_history)} actions")
        
        # Reset environment
        self.env.seed(self.episode_seed)
        self.observation = self.env.reset()
        self.current_time = self.env.current_timestamp
        
        # Clear history
        self.action_history = []
        self.current_step = 0
        self.done = False
        
        print(f"  - Environment reset: time={self.current_time/3600:.2f}h, queue_size={len(self.env.job_queue)}")
        print(f"  - Running jobs: {len(self.env.running_jobs)}")
        
        # Update visualization
        self._print_visual_state("AFTER RESET")
        self._update_visualization()
        
        print(f"✅ Episode reset with seed {self.episode_seed}")

    def step_episode(self, event=None):
        """Take one neural network action step forward in the episode"""
        print(f"\n➡️ NEURAL NETWORK ACTION STEP {self.current_step}")
        self._print_visual_state("BEFORE NEURAL ACTION")
        
        if self.done:
            print("🏁 Episode is completed")
            return
            
        try:
            # Store pre-action state for comparison
            pre_action_queue_size = len(self.env.job_queue)
            pre_action_running_size = len(self.env.running_jobs)
            pre_action_time = self.current_time
            
            # Build action masks
            mask1, mask2, mask3 = self._build_action_masks()
            print(f"  🎯 Neural Network Decision:")
            print(f"    - Queue size: {pre_action_queue_size}")
            print(f"    - Running jobs: {pre_action_running_size}")
            print(f"    - Action masks: mask1={sum(mask1)} available, mask2={sum(mask2)} available, mask3={sum(mask3)} available")
            
            # Get single neural network action
            action1, action2, action3 = self.model.eval_action(self.observation, mask1, mask2, mask3)
            
            # Convert to integers if they're tensors
            if hasattr(action1, 'item'):
                action1 = action1.item()
            if hasattr(action2, 'item'):
                action2 = action2.item()
            if hasattr(action3, 'item'):
                action3 = action3.item()
            
            # Save action to history
            self.action_history.append((action1, action2, action3))
            
            print(f"  🧠 Neural Network Output: Action1={action1}, Action2={action2}, Action3={action3}")
            
            # Execute single step in environment
            step_result = self.env.step(action1, action2, action3)
            
            # Update state
            self.observation = step_result[0]
            reward = step_result[1]
            self.done = step_result[2]
            self.current_time = self.env.current_timestamp
            self.current_step += 1
            
            # Analyze what changed
            post_action_queue_size = len(self.env.job_queue)
            post_action_running_size = len(self.env.running_jobs)
            time_advancement = self.current_time - pre_action_time
            
            print(f"  📊 Environment Changes:")
            print(f"    - Queue: {pre_action_queue_size} → {post_action_queue_size} (Δ{post_action_queue_size - pre_action_queue_size})")
            print(f"    - Running: {pre_action_running_size} → {post_action_running_size} (Δ{post_action_running_size - pre_action_running_size})")
            print(f"    - Time: {pre_action_time/3600:.2f}h → {self.current_time/3600:.2f}h (Δ{time_advancement/3600:.2f}h)")
            print(f"    - Reward: {reward:.3f}")
            print(f"    - Done: {self.done}")
            
            # Check for multiple simultaneous changes (which we want to avoid)
            queue_change = abs(post_action_queue_size - pre_action_queue_size)
            running_change = abs(post_action_running_size - pre_action_running_size)
            
            if queue_change > 1:
                print(f"  ⚠️  WARNING: Queue changed by {queue_change} jobs in single step!")
            if running_change > 1:
                print(f"  ⚠️  WARNING: Running jobs changed by {running_change} jobs in single step!")
            if time_advancement > 0 and running_change > 0:
                print(f"  ⚠️  WARNING: Both time advanced AND jobs changed in single step!")
            
            # Update visualization
            self._print_visual_state("AFTER NEURAL ACTION")
            self._update_visualization()
            
            return {
                'action1': action1, 'action2': action2, 'action3': action3,
                'reward': reward, 'done': self.done,
                'queue_change': queue_change,
                'running_change': running_change,
                'time_advancement': time_advancement
            }
            
        except Exception as e:
            print(f"❌ Error in neural network step: {e}")
            import traceback
            traceback.print_exc()

    def step_back(self, event=None):
        """Go back one neural network action step using reset-and-replay"""
        print(f"\n⬅️ NEURAL NETWORK ACTION STEP BACK from step {self.current_step}")
        print(f"  - Current action history length: {len(self.action_history)}")
        
        if self.current_step <= 0:
            print("🚫 Already at the beginning of episode")
            return
            
        try:
            # Remove the last action to get target state
            if not self.action_history:
                print("🚫 No actions to remove")
                return
                
            removed_action = self.action_history.pop()
            target_step = len(self.action_history)
            print(f"  - Removed neural network action: {removed_action}")
            print(f"  - Target step after removal: {target_step}")
            
            # Reset environment to beginning
            print(f"  - Resetting environment and replaying {target_step} actions...")
            self.env.seed(self.episode_seed)
            self.observation = self.env.reset()
            self.current_time = self.env.current_timestamp
            self.current_step = 0
            self.done = False
            
            print(f"  - Environment reset: time={self.current_time/3600:.2f}h, queue_size={len(self.env.job_queue)}")
            
            # Replay all remaining actions
            for step_idx, action in enumerate(self.action_history):
                print(f"    - Replaying step {step_idx}: {action}")
                
                # Build action masks for this replay step
                mask1, mask2, mask3 = self._build_action_masks()
                
                # Execute the stored action (not asking neural network again)
                step_result = self.env.step(action[0], action[1], action[2])
                
                # Update state
                self.observation = step_result[0]
                self.done = step_result[2]
                self.current_time = self.env.current_timestamp
                self.current_step += 1
                
                print(f"      → Time: {self.current_time/3600:.2f}h, Queue: {len(self.env.job_queue)}, Running: {len(self.env.running_jobs)}")
            
            print(f"⬅️ Successfully went back to neural step {self.current_step}")
            print(f"  - Final state: time={self.current_time/3600:.2f}h, queue={len(self.env.job_queue)}, running={len(self.env.running_jobs)}")
            
            self._print_visual_state("AFTER RESET-AND-REPLAY BACK")
            self._update_visualization()
                
        except Exception as e:
            print(f"❌ Error going back: {e}")
            import traceback
            traceback.print_exc()

    def toggle_auto_play(self, event=None):
        """Toggle automatic episode playback"""
        self.auto_play = not self.auto_play
        self.btn_auto.label.set_text('Pause' if self.auto_play else 'Auto')
        
        if self.auto_play:
            self._auto_play_loop()
        else:
            if self.auto_play_timer:
                try:
                    self.auto_play_timer.stop()
                except:
                    pass

    def _auto_play_loop(self):
        """Automatic playback loop"""
        if self.auto_play and not self.done:
            self.step_episode()
            self.play_speed = self.slider_speed.val
            
            try:
                import matplotlib.animation as animation
                self.auto_play_timer = self.fig.canvas.new_timer(interval=int(1000 / self.play_speed))
                self.auto_play_timer.single_shot = True
                self.auto_play_timer.add_callback(self._auto_play_loop)
                self.auto_play_timer.start()
            except Exception as e:
                print(f"Auto-play timer error: {e}")
                self.auto_play = False
                self.btn_auto.label.set_text('Auto')



    def _print_visual_state(self, context=""):
        """Print detailed state of all visual components for debugging"""
        print(f"📊 VISUAL STATE DEBUG {context}:")
        print(f"  📈 Core State:")
        print(f"    - Current step: {self.current_step}")
        print(f"    - Current time: {self.current_time/3600:.2f}h")
        print(f"    - Done: {self.done}")
        print(f"    - Episode seed: {self.episode_seed}")
        
        print(f"  📋 Action & History:")
        print(f"    - Action history length: {len(self.action_history)}")
        if len(self.action_history) > 0:
            print(f"    - Last action: {self.action_history[-1]}")
        
        print(f"  🏭 Environment State:")
        print(f"    - Environment timestamp: {self.env.current_timestamp/3600:.2f}h")
        print(f"    - Job queue size: {len(self.env.job_queue)}")
        print(f"    - Running jobs count: {len(self.env.running_jobs)}")
        if hasattr(self.env, 'next_arriving_job_idx'):
            print(f"    - Next arriving job idx: {self.env.next_arriving_job_idx}")
        
        print(f"  🔢 Job Details:")
        if len(self.env.job_queue) > 0:
            next_jobs = [job.job_id for job in self.env.job_queue[:5]]
            print(f"    - Next 5 jobs in queue: {next_jobs}")
        else:
            print(f"    - Job queue is empty")
            
        if len(self.env.running_jobs) > 0:
            running_ids = [job.job_id for job in self.env.running_jobs]
            print(f"    - Running job IDs: {running_ids}")
        else:
            print(f"    - No jobs currently running")
        
        # Count finished jobs
        finished_count = 0
        for job in self.env.loads.all_jobs:
            if hasattr(job, 'scheduled_time') and job.scheduled_time != -1:
                job_end_time = job.scheduled_time + job.run_time
                if job_end_time <= self.current_time:
                    finished_count += 1
        print(f"    - Finished jobs count: {finished_count}")
        
        print(f"  🎮 UI Controls:")
        print(f"    - Auto-play active: {getattr(self, 'auto_play', False)}")
        print(f"    - Play speed: {getattr(self, 'play_speed', 1.0)}")
        
        print(f"  📊 Visual Components Status:")
        axes_status = []
        for attr_name in ['ax_timeline', 'ax_carbon', 'ax_queue', 'ax_finished', 
                         'ax_stats', 'ax_action', 'ax_running']:
            if hasattr(self, attr_name):
                ax = getattr(self, attr_name)
                axes_status.append(f"{attr_name}: visible={ax.get_visible()}")
        print(f"    - Axes: {', '.join(axes_status)}")
        print("=" * 60)

    def _build_action_masks(self):
        """Build action masks for current observation"""
        MAX_QUEUE_SIZE = int(self.config.get('GAS-MARL setting', 'max_queue_size'))
        delayMaxJobNum = int(self.config.get('GAS-MARL setting', 'delaymaxjobnum'))
        action2_num = delayMaxJobNum + 8
        
        # Build masks based on current environment state
        mask1 = [1 if i >= len(self.env.job_queue) else 0 for i in range(MAX_QUEUE_SIZE)]
        
        # Build mask2 for delay actions
        mask2 = [0] * action2_num
        if len(self.env.running_jobs) == 0:
            # If no running jobs, mask out delay actions 1-20
            for i in range(1, delayMaxJobNum + 1):
                mask2[i] = 1
        else:
            # Mask delay actions based on number of running jobs
            num_running = len(self.env.running_jobs)
            for i in range(1, delayMaxJobNum + 1):
                if i > num_running:
                    mask2[i] = 1
        
        mask3 = [1 if i >= len(self.env.job_queue) else 0 for i in range(MAX_QUEUE_SIZE)]
        
        return mask1, mask2, mask3

    def _update_visualization(self):
        """Update all visualization components"""
        print(f"🎨 UPDATING VISUALIZATION at step {self.current_step}")
        
        self._clear_plots()
        print("  - Cleared all plots")
        
        self._plot_timeline()
        print("  - Updated timeline")
        
        self._plot_carbon_intensity()
        print("  - Updated carbon intensity")
        
        self._plot_job_queue()
        print("  - Updated job queue")
        
        self._plot_finished_jobs()
        print("  - Updated finished jobs")
        
        self._plot_statistics()
        print("  - Updated statistics")
        
        self._plot_current_action()
        print("  - Updated current action")
        
        self._plot_running_jobs()
        print("  - Updated running jobs")
        
        # Refresh display
        self.fig.canvas.draw()
        plt.pause(0.01)
        print("  - Display refreshed")

    def _clear_plots(self):
        """Clear all plot areas"""
        self.ax_timeline.clear()
        self.ax_carbon.clear()
        self.ax_queue.clear()
        self.ax_finished.clear()
        self.ax_stats.clear()
        self.ax_action.clear()
        self.ax_running.clear()
        
        # Reset titles
        self.ax_timeline.set_title('Job Scheduling Timeline', fontweight='bold')
        self.ax_carbon.set_title('Carbon Intensity', fontweight='bold')
        self.ax_queue.set_title('Job Queue', fontweight='bold')
        self.ax_finished.set_title('Recently Finished Jobs', fontweight='bold')
        self.ax_stats.set_title('Episode Statistics', fontweight='bold')
        self.ax_action.set_title('Current Action', fontweight='bold')
        self.ax_running.set_title('Currently Running Jobs', fontweight='bold')
        
        # Turn off axes for text panels
        self.ax_queue.axis('off')
        self.ax_finished.axis('off')
        self.ax_stats.axis('off')
        self.ax_action.axis('off')
        self.ax_running.axis('off')

    def _plot_timeline(self):
        """Plot the job scheduling timeline"""
        self.ax_timeline.set_title('Job Scheduling Timeline', fontweight='bold')
        self.ax_timeline.set_xlabel('Time (hours)')
        self.ax_timeline.set_ylabel('Processor Slots')
        
        max_processors = 256
        
        # Get all scheduled jobs
        finished_jobs = []
        running_jobs = []
        
        # Find all jobs that have been scheduled
        for job in self.env.loads.all_jobs:
            if hasattr(job, 'scheduled_time') and job.scheduled_time != -1:
                job_end_time = job.scheduled_time + job.run_time
                if job_end_time <= self.current_time:
                    finished_jobs.append(job)
                else:
                    running_jobs.append(job)
        
        # Plot finished jobs (gray with hatching)
        for job in finished_jobs:
            start_time = job.scheduled_time / 3600
            duration = job.run_time / 3600
            num_procs = len(job.allocated_machines) if job.allocated_machines else job.request_number_of_processors
            
            rect = patches.Rectangle(
                (start_time, 0), duration, num_procs,
                facecolor='lightgray', edgecolor='gray', alpha=0.6,
                hatch='///'
            )
            self.ax_timeline.add_patch(rect)
            
            self.ax_timeline.text(
                start_time + duration/2, num_procs/2,
                f'J{job.job_id}',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='darkgray'
            )
        
        # Plot running jobs (colorful)
        for job in running_jobs:
            start_time = job.scheduled_time / 3600
            duration = job.run_time / 3600
            num_procs = len(job.allocated_machines) if job.allocated_machines else job.request_number_of_processors
            
            color = self.job_colors[job.job_id % len(self.job_colors)]
            rect = patches.Rectangle(
                (start_time, 0), duration, num_procs,
                facecolor=color, edgecolor='black', alpha=0.8
            )
            self.ax_timeline.add_patch(rect)
            
            self.ax_timeline.text(
                start_time + duration/2, num_procs/2,
                f'J{job.job_id}',
                ha='center', va='center', fontsize=8, fontweight='bold'
            )
        
        # Plot current time line
        current_time_hours = self.current_time / 3600
        self.ax_timeline.axvline(
            current_time_hours, color='red', linestyle='--', linewidth=2,
            label='Current Time'
        )
        
        # Set axis limits based on jobs and current time
        all_jobs = finished_jobs + running_jobs
        if all_jobs:
            job_times = [job.scheduled_time / 3600 for job in all_jobs]
            job_end_times = [(job.scheduled_time + job.run_time) / 3600 for job in all_jobs]
            
            earliest_time = min(min(job_times), current_time_hours - 1)
            latest_time = max(max(job_end_times), current_time_hours + 4)
            
            self.ax_timeline.set_xlim(max(0, earliest_time), latest_time)
        else:
            self.ax_timeline.set_xlim(max(0, current_time_hours - 2), current_time_hours + 4)
        
        self.ax_timeline.set_ylim(0, max_processors)
        
        # Add legend
        legend_elements = [
            patches.Patch(color='lightgray', hatch='///', label=f'Finished Jobs ({len(finished_jobs)})'),
            patches.Patch(color='lightblue', label=f'Running Jobs ({len(running_jobs)})'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Current Time')
        ]
        self.ax_timeline.legend(handles=legend_elements, loc='upper right', fontsize=8)
        self.ax_timeline.grid(True, alpha=0.3)

    def _plot_carbon_intensity(self):
        """Plot carbon intensity over time"""
        self.ax_carbon.set_title('Carbon Intensity (72h window)', fontweight='bold')
        self.ax_carbon.set_xlabel('Simulation Time (hours)')
        self.ax_carbon.set_ylabel('gCO2eq/kWh')
        
        if hasattr(self.env.cluster, 'carbonIntensity'):
            carbon_data = self.env.cluster.carbonIntensity.carbonIntensityList
            current_hour_sim = self.current_time / 3600  # Actual simulation time in hours
            
            # Generate 72-hour window around current time
            window = 72
            start_hour_sim = current_hour_sim - window//2
            end_hour_sim = current_hour_sim + window//2
            
            # Create arrays for the 72-hour window
            sim_hours = np.arange(start_hour_sim, end_hour_sim + 1)
            intensities = []
            
            for hour in sim_hours:
                # Map simulation hour to carbon data index with year wraparound
                data_index = int(hour) % len(carbon_data)
                intensities.append(carbon_data[data_index])
            
            # Plot the carbon intensity
            self.ax_carbon.plot(sim_hours, intensities, color='green', linewidth=2)
            
            # Mark current time
            self.ax_carbon.axvline(
                current_hour_sim, color='red', linestyle='--', linewidth=2, label='Current Time'
            )
            
            # Highlight 24-hour forecast window
            forecast_end = current_hour_sim + 24
            self.ax_carbon.axvspan(
                current_hour_sim, forecast_end, alpha=0.2, color='orange', 
                label='24h Forecast Window'
            )
            
            # Set view limits
            self.ax_carbon.set_xlim(start_hour_sim, end_hour_sim)
            if intensities:
                margin = (max(intensities) - min(intensities)) * 0.1
                self.ax_carbon.set_ylim(
                    min(intensities) - margin,
                    max(intensities) + margin
                )
            
            # Add legend
            self.ax_carbon.legend(loc='upper right', fontsize=8)
        
        self.ax_carbon.grid(True, alpha=0.3)

    def _plot_job_queue(self):
        """Plot current job queue"""
        self.ax_queue.set_title('Job Queue', fontweight='bold')
        self.ax_queue.set_xlim(0, 10)
        self.ax_queue.set_ylim(0, 10)
        
        queue_jobs = self.env.job_queue[:10]  # Show top 10 jobs
        
        y_pos = 9
        for i, job in enumerate(queue_jobs):
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            
            # Determine job priority color
            if carbon_consideration > 0.7:
                color = self.priority_colors['high']
            elif carbon_consideration > 0.3:
                color = self.priority_colors['medium']
            else:
                color = self.priority_colors['low']
            
            # Draw job box
            rect = patches.Rectangle(
                (0.5, y_pos - 0.4), 9, 0.8,
                facecolor=color, edgecolor='black', alpha=0.7
            )
            self.ax_queue.add_patch(rect)
            
            # Add job info
            self.ax_queue.text(
                1, y_pos, f'J{job.job_id}',
                ha='left', va='center', fontsize=10, fontweight='bold'
            )
            self.ax_queue.text(
                5, y_pos, f'P:{job.request_number_of_processors} T:{job.run_time/3600:.1f}h',
                ha='center', va='center', fontsize=8
            )
            self.ax_queue.text(
                9, y_pos, f'C:{carbon_consideration:.2f}',
                ha='right', va='center', fontsize=8
            )
            
            y_pos -= 1
        
        self.ax_queue.axis('off')

    def _plot_finished_jobs(self):
        """Plot recently finished jobs"""
        self.ax_finished.set_title('Recently Finished Jobs', fontweight='bold')
        self.ax_finished.set_xlim(0, 10)
        self.ax_finished.set_ylim(0, 10)
        
        # Get finished jobs
        finished_jobs = []
        for job in self.env.loads.all_jobs:
            if hasattr(job, 'scheduled_time') and job.scheduled_time != -1:
                job_end_time = job.scheduled_time + job.run_time
                if job_end_time <= self.current_time:
                    finished_jobs.append(job)
        
        # Sort by completion time (most recent first)
        finished_jobs.sort(key=lambda job: job.scheduled_time + job.run_time, reverse=True)
        
        # Show top 10 most recently finished jobs
        recent_finished = finished_jobs[:10]
        
        y_pos = 9
        for i, job in enumerate(recent_finished):
            completion_time = job.scheduled_time + job.run_time
            time_since_completion = (self.current_time - completion_time) / 3600
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            
            # Color based on carbon consideration
            if carbon_consideration > 0.7:
                color = self.priority_colors['high']
            elif carbon_consideration > 0.3:
                color = self.priority_colors['medium']
            else:
                color = self.priority_colors['low']
            
            # Draw job box with fade based on how long ago it finished
            alpha = max(0.3, 1.0 - (time_since_completion / 24))  # Fade over 24 hours
            rect = patches.Rectangle(
                (0.5, y_pos - 0.4), 9, 0.8,
                facecolor=color, edgecolor='darkgray', alpha=alpha
            )
            self.ax_finished.add_patch(rect)
            
            # Add job info
            self.ax_finished.text(
                1, y_pos, f'J{job.job_id}',
                ha='left', va='center', fontsize=9, fontweight='bold'
            )
            self.ax_finished.text(
                4, y_pos, f'T:{job.run_time/3600:.1f}h',
                ha='center', va='center', fontsize=8
            )
            self.ax_finished.text(
                6.5, y_pos, f'C:{carbon_consideration:.2f}',
                ha='center', va='center', fontsize=8
            )
            self.ax_finished.text(
                9, y_pos, f'-{time_since_completion:.1f}h',
                ha='right', va='center', fontsize=7, style='italic'
            )
            
            y_pos -= 1
        
        # Add summary text at bottom
        if finished_jobs:
            self.ax_finished.text(
                5, 0.5, f'Total Finished: {len(finished_jobs)}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8)
            )
        
        self.ax_finished.axis('off')

    def _plot_statistics(self):
        """Plot episode statistics"""
        self.ax_stats.set_title('Episode Statistics', fontweight='bold')
        
        # Calculate statistics
        total_jobs = self.env.loads.size()
        finished_jobs = sum(1 for job in self.env.loads.all_jobs
                          if hasattr(job, 'scheduled_time') and job.scheduled_time != -1
                          and job.scheduled_time + job.run_time <= self.current_time)
        running_jobs = len(self.env.running_jobs)
        queue_size = len(self.env.job_queue)
        total_scheduled = finished_jobs + running_jobs
        
        # Current utilization
        total_procs = 256
        used_procs = sum(len(job.allocated_machines) if job.allocated_machines
                        else job.request_number_of_processors
                        for job in self.env.running_jobs)
        utilization = (used_procs / total_procs) * 100 if total_procs > 0 else 0
        
        stats_text = f"""Step: {self.current_step}
Current Time: {self.current_time/3600:.2f}h
Jobs Scheduled: {total_scheduled}/{total_jobs}
Finished Jobs: {finished_jobs}
Running Jobs: {running_jobs}
Queue Size: {queue_size}
Utilization: {utilization:.1f}%"""
        
        self.ax_stats.text(
            0.05, 0.95, stats_text,
            transform=self.ax_stats.transAxes,
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8)
        )
        
        self.ax_stats.axis('off')

    def _plot_current_action(self):
        """Plot information about the current neural network action"""
        self.ax_action.set_title('Neural Network Action', fontweight='bold')
        
        if self.action_history:
            last_action = self.action_history[-1]
            
            action_text = f"""🧠 Neural Step {len(self.action_history)}
Job Select: {last_action[0]}
Delay: {last_action[1]}
Backfill: {last_action[2]}"""
            
            # Add interpretation of actions
            if last_action[0] < len(self.env.job_queue):
                selected_job = self.env.job_queue[last_action[0]]
                action_text += f"\n\n📋 Selected Job {selected_job.job_id}"
                action_text += f"\nProcs: {selected_job.request_number_of_processors}"
                action_text += f"\nTime: {selected_job.run_time/3600:.1f}h"
                if hasattr(selected_job, 'carbon_consideration'):
                    action_text += f"\nCarbon: {selected_job.carbon_consideration:.2f}"
            else:
                action_text += f"\n\n⏸️ No job selected"
            
            # Determine backfill type
            if self.backfill == 0:
                backfill_type = "None"
            elif self.backfill == 1:
                backfill_type = "Carbon-aware"
            elif self.backfill == 2:
                backfill_type = "EASY"
            elif self.backfill == 3:
                backfill_type = "Neural"
            else:
                backfill_type = f"Mode {self.backfill}"
            
            action_text += f"\n\nBackfill: {backfill_type}"
            
        else:
            action_text = "🎯 Waiting for first\nneural network action..."
        
        self.ax_action.text(
            0.05, 0.95, action_text,
            transform=self.ax_action.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
        )
        
        self.ax_action.axis('off')

    def _plot_running_jobs(self):
        """Plot currently running jobs"""
        self.ax_running.set_title('Currently Running Jobs', fontweight='bold')
        self.ax_running.set_xlim(0, 10)
        self.ax_running.set_ylim(0, 5)
        
        running_jobs = self.env.running_jobs[:50]  # Show up to 50 running jobs (more space now)
        
        x_pos = 0.5
        y_pos = 4.5
        jobs_per_row = 10
        
        for i, job in enumerate(running_jobs):
            if i > 0 and i % jobs_per_row == 0:
                y_pos -= 1
                x_pos = 0.5
            
            # Calculate remaining time
            remaining_time = (job.scheduled_time + job.run_time - self.current_time) / 3600
            
            # Color based on remaining time
            if remaining_time > 2:
                color = self.priority_colors['low']  # Blue - long time left
            elif remaining_time > 0.5:
                color = self.priority_colors['medium']  # Teal - medium time
            else:
                color = self.priority_colors['high']  # Red - almost done
            
            # Draw job box
            rect = patches.Rectangle(
                (x_pos, y_pos - 0.2), 0.8, 0.4,
                facecolor=color, edgecolor='black', alpha=0.7
            )
            self.ax_running.add_patch(rect)
            
            # Add job ID and remaining time
            self.ax_running.text(
                x_pos + 0.4, y_pos + 0.1, f'J{job.job_id}',
                ha='center', va='center', fontsize=8, fontweight='bold'
            )
            self.ax_running.text(
                x_pos + 0.4, y_pos - 0.1, f'{remaining_time:.1f}h',
                ha='center', va='center', fontsize=6, style='italic'
            )
            
            x_pos += 1
        
        self.ax_running.axis('off')

    def run(self):
        """Start the interactive visualization"""
        print("🎮 Starting Interactive Neural Network Action Visualizer")
        print("=" * 70)
        print("How it works:")
        print("  - Reset: Resets the episode to the beginning")
        print("  - Step: Execute ONE neural network action (not full scheduling)")
        print("  - Back: Go back one neural network action (using saved history)")
        print("  - Auto: Automatically step through individual neural actions")
        print("  - Speed: Control auto-play speed")
        print()
        print("Each step shows ONE neural network decision and its immediate effect!")
        print("You'll see individual job scheduling/timing decisions, not batches.")
        print("=" * 70)
        
        # Perform initial reset
        print("\n🔄 Initializing episode...")
        self.reset_episode()
        
        plt.show(block=True)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Interactive Episode Scheduling Visualizer')
    parser.add_argument('model_path', help='Path to model directory')
    parser.add_argument('--epoch', type=int, help='Epoch number to load')
    parser.add_argument('--workload', default='lublin_256_carbon', help='Workload to visualize')
    parser.add_argument('--backfill', type=int, default=3, help='Backfill mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Determine actor weights path
    if args.epoch is not None:
        actor_weights_path = os.path.join(args.model_path, f"checkpoints/epoch_{args.epoch}")
    else:
        actor_weights_path = os.path.join(args.model_path, "final")
    
    # Create and run visualizer
    try:
        visualizer = SchedulingVisualizer(
            actor_weights_path=actor_weights_path,
            workload=args.workload,
            backfill=args.backfill,
            debug=args.debug
        )
        visualizer.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 