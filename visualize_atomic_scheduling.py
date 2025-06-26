#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import argparse
import os
import sys
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from HPCSimPickJobs import HPCEnv
from atomic_hpc_env import AtomicHPCEnv, AtomicStepType
import configparser

# Try to import torch and models (needed for TorchModelWrapper)
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    print("⚠️ PyTorch not available - only legacy pickle models will work")

# Load configuration
config = configparser.ConfigParser()
config.read('configFile/config.ini')

MAX_QUEUE_SIZE = int(config.get('GAS-MARL setting', 'max_queue_size'))
MAX_POWER = float(config.get('algorithm constants', 'max_power'))
MAX_CARBON_INTENSITY = float(config.get('algorithm constants', 'max_carbon_intensity'))

class TorchModelWrapper:
    """Wrapper for PyTorch state dict models"""
    def __init__(self, model_path, config_path=None, device='cpu'):
        import torch
        
        # Load configuration to get architecture parameters
        config = configparser.ConfigParser()
        if config_path and os.path.exists(config_path):
            config.read(config_path)
            print(f"📋 Using experiment config: {config_path}")
        else:
            config.read('configFile/config.ini')
            print(f"📋 Using global config: configFile/config.ini")
        
        # Get architecture parameters from config
        MAX_QUEUE_SIZE = int(config.get('GAS-MARL setting', 'max_queue_size'))
        run_win = int(config.get('GAS-MARL setting', 'run_win'))
        green_win = int(config.get('GAS-MARL setting', 'green_win'))
        JOB_FEATURES = int(config.get('algorithm constants', 'job_features'))
        RUN_FEATURE = int(config.get('algorithm constants', 'run_feature'))
        GREEN_FEATURE = int(config.get('algorithm constants', 'green_feature'))
        
        # Also get delay parameters for action2_num calculation
        delayMaxJobNum = int(config.get('GAS-MARL setting', 'delaymaxjobnum'))
        delayTimeList = eval(config.get('GAS-MARL setting', 'delaytimelist'))
        self.expected_action2_size = len(delayTimeList) + delayMaxJobNum + 1
        
        print(f"🔧 Expected action2 size: {self.expected_action2_size} (delayMaxJobNum={delayMaxJobNum}, delayTimeList length={len(delayTimeList)})")
        
        inputNum_size = [MAX_QUEUE_SIZE, run_win, green_win]
        featureNum_size = [JOB_FEATURES, RUN_FEATURE, GREEN_FEATURE]
        
        # Try different PPO implementations to find the right one
        self.ppo = None
        self.model_type = None
        
        # Try MARL.py first (2-action implementation)
        try:
            from MARL import PPO
            self.ppo = PPO(
                batch_size=256,
                inputNum_size=inputNum_size,
                featureNum_size=featureNum_size,
                device=device
            )
            
            state_dict = torch.load(model_path, map_location=device)
            self.ppo.actor_net.load_state_dict(state_dict)
            self.ppo.actor_net.eval()
            self.model_type = "MARL_2action"
            print(f"✅ Loaded 2-action MARL model from {model_path}")
            
        except Exception as marl_error:
            print(f"MARL.py loading failed: {marl_error}")
            
            # Try MARL_backfill.py (3-action implementation)
            try:
                from MARL_backfill import PPO
                self.ppo = PPO(
                    batch_size=256,
                    inputNum_size=inputNum_size,
                    featureNum_size=featureNum_size,
                    device=device,
                    backfill=3
                )
                
                state_dict = torch.load(model_path, map_location=device)
                self.ppo.actor_net.load_state_dict(state_dict)
                self.ppo.actor_net.eval()
                self.model_type = "MARL_3action"
                print(f"✅ Loaded 3-action MARL_backfill model from {model_path}")
                
            except Exception as backfill_error:
                print(f"MARL_backfill.py loading failed: {backfill_error}")
                raise ValueError(f"Could not load model with either MARL.py or MARL_backfill.py architectures")
        
        self.device = device
        
    def eval_action(self, observation, mask1, mask2, mask3):
        import numpy as np
        
        # Convert masks to numpy arrays if they're lists
        if isinstance(mask1, list):
            mask1 = np.array(mask1)
        if isinstance(mask2, list):
            mask2 = np.array(mask2)
        if isinstance(mask3, list):
            mask3 = np.array(mask3)
        
        if self.model_type == "MARL_2action":
            # Use the expected action2 size from the experiment config
            expected_action2_size = self.expected_action2_size
            
            # If mask2 is smaller than expected, pad it with 1s (masked out)
            if len(mask2) < expected_action2_size:
                padding_size = expected_action2_size - len(mask2)
                mask2_padded = np.concatenate([mask2, np.ones(padding_size)])
                print(f"🔧 Padded mask2 from {len(mask2)} to {len(mask2_padded)} elements")
                mask2 = mask2_padded
            # If mask2 is larger than expected, truncate it
            elif len(mask2) > expected_action2_size:
                mask2 = mask2[:expected_action2_size]
                print(f"🔧 Truncated mask2 to {expected_action2_size} elements")
            
            # Use 2-action version
            action1, action2 = self.ppo.eval_action(observation, mask1, mask2)
            return action1, action2, 0  # Return 0 for third action
        elif self.model_type == "MARL_3action":
            # Use 3-action version
            return self.ppo.eval_action(observation, mask1, mask2, mask3)
        else:
            raise ValueError("Unknown model type")

class LegacyModelWrapper:
    """Wrapper for legacy pickled PPO models"""
    def __init__(self, old_ppo):
        self.ppo = old_ppo
        
    def eval_action(self, observation, mask1, mask2, mask3):
        # Convert numpy arrays to expected format
        obs_tensor = np.array(observation).reshape(1, -1)
        mask1_tensor = np.array(mask1).reshape(1, -1) 
        mask2_tensor = np.array(mask2).reshape(1, -1)
        mask3_tensor = np.array(mask3).reshape(1, -1)
        
        # Get action from old PPO format
        try:
            action1, action2, action3 = self.ppo.choose_action(
                obs_tensor, mask1_tensor, mask2_tensor, mask3_tensor
            )
        except TypeError:
            # Try 2-action version
            action1, action2 = self.ppo.choose_action(obs_tensor, mask1_tensor, mask2_tensor)
            action3 = 0
        
        # Handle different return types
        if hasattr(action1, 'cpu'):
            action1 = action1.cpu().numpy()[0]
        if hasattr(action2, 'cpu'):
            action2 = action2.cpu().numpy()[0]
        if hasattr(action3, 'cpu'):
            action3 = action3.cpu().numpy()[0]
        
        return action1, action2, action3

class AtomicSchedulingVisualizer:
    """
    Interactive visualizer for atomic HPC scheduling steps.
    Shows individual events clearly: job arrivals, completions, neural decisions.
    """
    
    def __init__(self, actor_weights_path, workload="lublin_256_carbon_float_simple", backfill=3, debug=False):
        self.actor_weights_path = actor_weights_path
        self.workload = workload
        self.backfill = backfill
        self.debug = debug
        
        # Extract experiment directory to find config_snapshot.ini
        self.experiment_dir = self._extract_experiment_dir(actor_weights_path)
        self.config_path = None
        if self.experiment_dir:
            potential_config = os.path.join(self.experiment_dir, 'config_snapshot.ini')
            if os.path.exists(potential_config):
                self.config_path = potential_config
        
        # Initialize environment
        self._setup_environment()
        self._load_model()
        
        # Visualization state
        self.current_step = 0
        self.current_time = 0
        self.done = False
        self.episode_seed = 42
        
        # State history for proper back functionality
        self.state_history = []  # Store complete environment states
        self.event_history = []  # Store event descriptions for display
        
        # Enhanced color scheme for carbon visualization
        self.setup_color_schemes()
        
        # Setup visualization
        self._setup_plots()
        self._setup_controls()
        
        # Initialize episode
        self.reset_episode()
    
    def setup_color_schemes(self):
        """Setup enhanced color schemes for carbon visualization"""
        import matplotlib.colors as mcolors
        
        # Create red-to-green colormap for carbon consideration
        # 0 = red (low carbon awareness), 1 = bright green (high carbon awareness)
        self.carbon_consideration_cmap = mcolors.LinearSegmentedColormap.from_list(
            "red_to_green", 
            [(0, "#cc0000"),      # Red
             (0.3, "#cc4400"),    # Red-orange
             (0.5, "#cccc00"),    # Yellow
             (0.7, "#66cc00"),    # Yellow-green  
             (1.0, "#00cc00")]    # Green
        )
        
        # Create colormap for actual carbon emissions (blue to red)
        # Low emissions = blue, high emissions = red
        self.carbon_emissions_cmap = mcolors.LinearSegmentedColormap.from_list(
            "blue_to_red",
            [(0, "#0066cc"),      # Blue (low emissions)
             (0.5, "#ffff00"),    # Yellow (medium emissions) 
             (1.0, "#cc0000")]    # Red (high emissions)
        )
        
        # Legacy color scheme for backward compatibility
        self.priority_colors = {
            'high': '#ff7f7f',     # Light red for high carbon consideration
            'medium': '#ffff7f',   # Light yellow for medium
            'low': '#7fff7f'       # Light green for low
        }
    
    def get_carbon_consideration_color(self, carbon_consideration):
        """Get color based on carbon consideration value (0-1)"""
        # Clamp to valid range
        carbon_consideration = max(0, min(1, carbon_consideration))
        return self.carbon_consideration_cmap(carbon_consideration)
    
    def get_carbon_emissions_color(self, actual_emissions, max_emissions):
        """Get color based on actual carbon emissions relative to maximum"""
        if max_emissions == 0:
            return self.carbon_emissions_cmap(0)
        
        emission_ratio = actual_emissions / max_emissions
        emission_ratio = max(0, min(1, emission_ratio))
        return self.carbon_emissions_cmap(emission_ratio)
    
    def calculate_job_carbon_emissions(self, job):
        """Calculate actual carbon emissions for a job"""
        if not hasattr(job, 'scheduled_time') or job.scheduled_time == -1:
            return 0, 0  # Not scheduled yet
            
        if not hasattr(self.env.base_env.cluster, 'carbonIntensity'):
            return 0, 0  # No carbon data
            
        # Get carbon intensity during job execution
        carbon_data = self.env.base_env.cluster.carbonIntensity.carbonIntensityList
        
        start_hour = int(job.scheduled_time / 3600)
        end_hour = int((job.scheduled_time + job.run_time) / 3600)
        
        total_emissions = 0
        power_kw = job.power / 1000.0  # Convert watts to kW
        
        for hour in range(start_hour, end_hour + 1):
            carbon_intensity = carbon_data[hour % len(carbon_data)]  # gCO2/kWh
            time_fraction = min(1.0, (job.run_time - (hour - start_hour) * 3600) / 3600)
            time_fraction = max(0, time_fraction)
            
            emissions = carbon_intensity * power_kw * time_fraction  # gCO2
            total_emissions += emissions
            
        # Also calculate worst-case emissions (highest carbon intensity)
        max_carbon_intensity = max(carbon_data)
        worst_case_emissions = max_carbon_intensity * power_kw * (job.run_time / 3600)
        
        return total_emissions, worst_case_emissions
    
    def _extract_experiment_dir(self, actor_weights_path):
        """Extract experiment directory from actor weights path"""
        # Examples:
        # lublin_256_carbon_float_simple/MARL_direct_co2_no_wait_no_backfill_incread_lr/checkpoints/epoch_20/_actor.pkl
        # -> lublin_256_carbon_float_simple/MARL_direct_co2_no_wait_no_backfill_incread_lr
        
        path_parts = actor_weights_path.split(os.path.sep)
        
        # Look for the experiment directory pattern
        # It should be before 'checkpoints' or 'final'
        for i, part in enumerate(path_parts):
            if part in ['checkpoints', 'final']:
                # The experiment dir is everything up to this part
                return os.path.join(*path_parts[:i])
        
        # If no checkpoints/final found, try to find the second-to-last directory
        if len(path_parts) >= 2:
            # For direct paths like 'experiment_dir/_actor.pkl'
            return os.path.dirname(actor_weights_path)
        
        return None
    
    def _setup_environment(self):
        """Initialize the HPC environment"""
        # Create base environment
        base_env = HPCEnv(backfill=self.backfill, debug=self.debug)
        base_env.my_init(workload_file=f"./data/{self.workload}.swf")
        
        # Wrap in atomic environment with experiment config
        self.env = AtomicHPCEnv(base_env, debug=self.debug, config_path=self.config_path)
        
        if self.config_path:
            print(f"🔧 Atomic environment using experiment config: {self.config_path}")
        else:
            print(f"🔧 Atomic environment using default config")
    
    def _load_model(self):
        """Load the neural network model"""
        print(f"Loading model from: {self.actor_weights_path}")
        if self.config_path:
            print(f"Using experiment config: {self.config_path}")
        else:
            print(f"No experiment config found, using global config")
        
        try:
            # First try loading as PyTorch state dict (newer format)
            import torch
            try:
                self.model = TorchModelWrapper(self.actor_weights_path, config_path=self.config_path, device='cpu')
                return
            except Exception as torch_error:
                print(f"PyTorch loading failed: {torch_error}")
                
            # Fall back to pickle loading (legacy format)
            try:
                with open(self.actor_weights_path, 'rb') as f:
                    loaded_model = pickle.load(f)
                
                # Check if it's a legacy PPO model
                if hasattr(loaded_model, 'choose_action') or hasattr(loaded_model, 'eval_action'):
                    self.model = LegacyModelWrapper(loaded_model)
                    print("✅ Loaded legacy pickled PPO model")
                    return
                else:
                    print("❌ Pickled model type not recognized")
                    raise ValueError("Unknown pickled model type")
                    
            except Exception as pickle_error:
                print(f"Pickle loading failed: {pickle_error}")
                raise ValueError(f"Could not load model as PyTorch state dict or pickle file")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def _setup_plots(self):
        """Setup the visualization plots"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.suptitle('Atomic HPC Scheduling Visualization', fontsize=16, fontweight='bold')
        
        # Create grid layout (4x4)
        gs = self.fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Main timeline (top, spans 2x4)
        self.ax_timeline = self.fig.add_subplot(gs[0:2, :])
        
        # Event info panel (middle left, 1x2)
        self.ax_event_info = self.fig.add_subplot(gs[2, 0:2])
        
        # Job queue (middle right, 1x1)
        self.ax_queue = self.fig.add_subplot(gs[2, 2])
        
        # Running jobs (middle right, 1x1)
        self.ax_running = self.fig.add_subplot(gs[2, 3])
        
        # Statistics (bottom left, 1x2)
        self.ax_stats = self.fig.add_subplot(gs[3, 0:2])
        
        # Carbon intensity (bottom right, 1x2)
        self.ax_carbon = self.fig.add_subplot(gs[3, 2:4])
        
        plt.tight_layout()
    
    def _setup_controls(self):
        """Setup control buttons"""
        # Add button axes
        ax_reset = plt.axes([0.02, 0.02, 0.08, 0.04])
        ax_next_event = plt.axes([0.12, 0.02, 0.12, 0.04])
        ax_neural_step = plt.axes([0.26, 0.02, 0.12, 0.04])
        ax_back = plt.axes([0.50, 0.02, 0.08, 0.04])
        
        # Create buttons
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_next_event = Button(ax_next_event, 'Next Event')
        self.btn_neural_step = Button(ax_neural_step, 'Neural Step')
        self.btn_back = Button(ax_back, 'Back')
        
        # Connect callbacks
        self.btn_reset.on_clicked(self.reset_episode)
        self.btn_next_event.on_clicked(self.step_next_event)
        self.btn_neural_step.on_clicked(self.step_neural_only)
        self.btn_back.on_clicked(self.step_back)
    
    def reset_episode(self, event=None):
        """Reset the episode"""
        print(f"\n🔄 RESETTING EPISODE (seed: {self.episode_seed})")
        
        # Reset environment
        self.env.base_env.seed(self.episode_seed)
        self.observation = self.env.reset()
        self.current_time = self.env.base_env.current_timestamp
        self.current_step = 0
        self.done = False
        
        # Clear history
        self.state_history = []
        self.event_history = []
        
        # Save initial state
        self._save_current_state()
        
        print(f"Episode reset: time={self.current_time/3600:.2f}h")
        
        # Update visualization
        self._update_visualization()
    
    def step_next_event(self, event=None):
        """Take the next atomic event step"""
        if self.done:
            print("🏁 Episode is completed")
            return
            
        # Get available events
        available_events = self.env.get_next_atomic_events()
        
        if not available_events:
            print("🏁 No more events - episode complete")
            self.done = True
            self._update_visualization()
            return
        
        # Take the next event (highest priority)
        next_event = available_events[0]
        event_type, description, priority = next_event
        
        print(f"\n⚡ ATOMIC EVENT #{self.current_step + 1}: {event_type.value}")
        print(f"  📝 {description}")
        
        # Execute the atomic step
        step_result = self._execute_atomic_step(event_type)
        
        # Update state
        self.current_step += 1
        self.current_time = step_result['timestamp']
        
        # Record event in history (for display)
        self.event_history.append({
            'step': self.current_step,
            'event_type': event_type,
            'description': description,
            'step_result': step_result,
            'timestamp': step_result['timestamp']
        })
        
        # Save the new state after this event
        self._save_current_state()
        
        # Update visualization
        self._update_visualization()
    
    def step_neural_only(self, event=None):
        """Take only neural network decision steps"""
        if self.done:
            print("🏁 Episode is completed")
            return
            
        # Look for neural decision events
        available_events = self.env.get_next_atomic_events()
        neural_events = [e for e in available_events if e[0] == AtomicStepType.NEURAL_DECISION]
        
        if not neural_events:
            print("🧠 No neural decisions available - take other events first")
            self.step_next_event()
            return
        
        # Execute neural decision
        step_result = self._execute_atomic_step(AtomicStepType.NEURAL_DECISION)
        
        # Update state
        self.current_step += 1
        self.current_time = step_result['timestamp']
        
        # Record in history (for display)
        self.event_history.append({
            'step': self.current_step,
            'event_type': AtomicStepType.NEURAL_DECISION,
            'description': step_result['description'],
            'step_result': step_result,
            'timestamp': step_result['timestamp']
        })
        
        # Save the new state after this event
        self._save_current_state()
        
        print(f"\n🧠 NEURAL DECISION #{self.current_step}")
        print(f"  📝 {step_result['description']}")
        
        # Update visualization
        self._update_visualization()
    
    def _execute_atomic_step(self, event_type):
        """Execute a specific atomic step"""
        if event_type == AtomicStepType.JOB_ARRIVAL:
            return self.env.step_job_arrival()
        elif event_type == AtomicStepType.JOB_COMPLETION:
            return self.env.step_job_completion()
        elif event_type == AtomicStepType.NEURAL_DECISION:
            return self.env.step_neural_decision(self.model)
        elif event_type == AtomicStepType.TIME_ADVANCE:
            return self.env.step_time_advance()
        else:
            return {'step_type': event_type, 'description': 'Unknown event', 'timestamp': self.current_time}
    
    def step_back(self, event=None):
        """Step back to previous event using state restoration"""
        if len(self.state_history) <= 1:  # Need at least 2 states (initial + 1 step)
            print("🚫 No events to step back from")
            return
        
        # Remove current state and event
        self.state_history.pop()  # Remove current state
        removed_event = self.event_history.pop()  # Remove current event
        
        # Restore previous state
        previous_state = self.state_history[-1]  # Get previous state (but keep it for future forward steps)
        self._restore_state(previous_state)
        
        print(f"\n⬅️ STEPPED BACK from event #{removed_event['step']}")
        print(f"  📝 Undid: {removed_event['description']}")
        print(f"  🔄 Restored to step #{self.current_step}")
        
        # Update visualization
        self._update_visualization()
    
    def _update_visualization(self):
        """Update all visualization panels"""
        self._clear_plots()
        self._plot_timeline()
        self._plot_event_info()
        self._plot_job_queue()
        self._plot_running_jobs()
        self._plot_statistics()
        self._plot_carbon_intensity()
        
        plt.draw()
    
    def _clear_plots(self):
        """Clear all plot axes"""
        for ax in [self.ax_timeline, self.ax_event_info, self.ax_queue, 
                   self.ax_running, self.ax_stats, self.ax_carbon]:
            ax.clear()
    
    def _plot_timeline(self):
        """Plot job timeline with enhanced carbon visualization"""
        self.ax_timeline.set_title('Job Timeline (Red→Green: Carbon Consideration, Blue→Red: Actual Emissions)', 
                                 fontweight='bold', fontsize=12)
        self.ax_timeline.set_xlabel('Time (hours)')
        self.ax_timeline.set_ylabel('Processors')
        
        current_time_hours = self.current_time / 3600
        max_processors = 256
        
        # Collect all finished jobs for emissions analysis
        finished_jobs = []
        max_actual_emissions = 0
        
        for job in self.env.base_env.loads.all_jobs:
            if (hasattr(job, 'scheduled_time') and job.scheduled_time != -1):
                end_time_job = job.scheduled_time + job.run_time
                if end_time_job <= self.current_time:
                    actual_emissions, worst_case_emissions = self.calculate_job_carbon_emissions(job)
                    finished_jobs.append((job, actual_emissions, worst_case_emissions))
                    max_actual_emissions = max(max_actual_emissions, actual_emissions)
        
        # Plot finished jobs with carbon emissions color coding
        for job, actual_emissions, worst_case_emissions in finished_jobs:
            start_time = job.scheduled_time / 3600
            end_time = (job.scheduled_time + job.run_time) / 3600
            processors = job.request_number_of_processors
            
            # Color by actual carbon emissions
            emissions_color = self.get_carbon_emissions_color(actual_emissions, max_actual_emissions)
            
            rect = patches.Rectangle(
                (start_time, 0), end_time - start_time, processors,
                facecolor=emissions_color, edgecolor='darkgray', alpha=0.7, linewidth=1
            )
            self.ax_timeline.add_patch(rect)
            
            # Add job info with emissions data
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            if end_time - start_time > 0.5:  # Only show text if job is wide enough
                self.ax_timeline.text(
                    start_time + (end_time - start_time) / 2, processors / 2,
                    f'J{job.job_id}\n{actual_emissions:.0f}g', 
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    color='white' if actual_emissions > max_actual_emissions * 0.5 else 'black'
                )
        
        # Plot running jobs with carbon consideration color coding
        for job in self.env.base_env.running_jobs:
            start_time = job.scheduled_time / 3600
            end_time = (job.scheduled_time + job.run_time) / 3600
            processors = len(job.allocated_machines) if job.allocated_machines else job.request_number_of_processors
            
            # Color by carbon consideration (black to green)
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            consideration_color = self.get_carbon_consideration_color(carbon_consideration)
            
            rect = patches.Rectangle(
                (start_time, 0), end_time - start_time, processors,
                facecolor=consideration_color, edgecolor='white', alpha=0.9, linewidth=2
            )
            self.ax_timeline.add_patch(rect)
            
            # Add job info
            remaining_time = (job.scheduled_time + job.run_time - self.current_time) / 3600
            if end_time - start_time > 0.5:  # Only show text if job is wide enough
                text_color = 'white' if carbon_consideration < 0.3 else 'black'
                self.ax_timeline.text(
                    start_time + (end_time - start_time) / 2, processors / 2,
                    f'J{job.job_id}\nC:{carbon_consideration:.2f}\n{remaining_time:.1f}h left',
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    color=text_color
                )
        
        # Current time line
        self.ax_timeline.axvline(current_time_hours, color='red', linestyle='--', linewidth=3,
                               label=f'Now: {current_time_hours:.1f}h')
        
        # Add legend for color coding
        legend_text = "🎨 Running Jobs: Carbon Consideration (C:)\n" \
                     "   🔴 Red = Low (0.0), 🟢 Green = High (1.0)\n" \
                     "🎨 Finished Jobs: Actual Emissions\n" \
                     "   🔵 Blue = Low, 🔴 Red = High"
        self.ax_timeline.text(0.02, 0.98, legend_text, transform=self.ax_timeline.transAxes,
                            fontsize=9, va='top', ha='left',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        
        # Set limits
        self.ax_timeline.set_xlim(max(0, current_time_hours - 4), current_time_hours + 8)
        self.ax_timeline.set_ylim(0, max_processors)
        self.ax_timeline.grid(True, alpha=0.3)
        self.ax_timeline.legend(loc='upper right')
    
    def _plot_event_info(self):
        """Plot current event information"""
        self.ax_event_info.set_title('Next Available Events', fontweight='bold')
        
        available_events = self.env.get_next_atomic_events()
        
        y_pos = 0.9
        for i, (event_type, description, priority) in enumerate(available_events[:5]):
            # Color code by event type
            if event_type == AtomicStepType.NEURAL_DECISION:
                color = 'lightblue'
            elif event_type == AtomicStepType.JOB_ARRIVAL:
                color = 'lightgreen'
            elif event_type == AtomicStepType.JOB_COMPLETION:
                color = 'lightyellow'
            else:
                color = 'lightgray'
            
            self.ax_event_info.text(
                0.05, y_pos, f"{i+1}. [{event_type.value}] {description}",
                transform=self.ax_event_info.transAxes,
                fontsize=10, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
            )
            y_pos -= 0.18
        
        if not available_events:
            self.ax_event_info.text(
                0.5, 0.5, "No events available\nEpisode Complete",
                transform=self.ax_event_info.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold'
            )
        
        self.ax_event_info.set_xlim(0, 1)
        self.ax_event_info.set_ylim(0, 1)
        self.ax_event_info.axis('off')
    
    def _plot_job_queue(self):
        """Plot current job queue with enhanced carbon visualization"""
        self.ax_queue.set_title(f'Job Queue ({len(self.env.base_env.job_queue)})\nC: = Carbon Consideration', 
                               fontweight='bold', fontsize=10)
        
        queue_jobs = self.env.base_env.job_queue[:10]  # Show top 10
        
        y_pos = 9
        for job in queue_jobs:
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            
            # Use red-to-green gradient
            color = self.get_carbon_consideration_color(carbon_consideration)
            text_color = 'white' if carbon_consideration < 0.3 else 'black'
            
            # Draw job box
            rect = patches.Rectangle(
                (0, y_pos - 0.4), 10, 0.8,
                facecolor=color, edgecolor='gray', alpha=0.8, linewidth=1
            )
            self.ax_queue.add_patch(rect)
            
            # Add job info with better formatting
            self.ax_queue.text(
                5, y_pos, f'J{job.job_id} | P:{job.request_number_of_processors} | C:{carbon_consideration:.2f}',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=text_color
            )
            
            y_pos -= 1
        
        # Add color legend at bottom
        if queue_jobs:
            self.ax_queue.text(5, 0.5, '🔴 Low C: → 🟢 High C:', 
                             ha='center', va='center', fontsize=8, style='italic')
        
        self.ax_queue.set_xlim(0, 10)
        self.ax_queue.set_ylim(0, 10)
        self.ax_queue.axis('off')
    
    def _plot_running_jobs(self):
        """Plot currently running jobs with enhanced carbon visualization"""
        self.ax_running.set_title(f'Running Jobs ({len(self.env.base_env.running_jobs)})\nCarbon Consideration', 
                                 fontweight='bold', fontsize=10)
        
        running_jobs = self.env.base_env.running_jobs[:10]  # Show top 10
        
        y_pos = 9
        for job in running_jobs:
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            remaining_time = (job.scheduled_time + job.run_time - self.current_time) / 3600
            
            # Use red-to-green gradient
            color = self.get_carbon_consideration_color(carbon_consideration)
            text_color = 'white' if carbon_consideration < 0.3 else 'black'
            
            # Draw job box with glowing border for running jobs
            rect = patches.Rectangle(
                (0, y_pos - 0.4), 10, 0.8,
                facecolor=color, edgecolor='orange', alpha=0.8, linewidth=2
            )
            self.ax_running.add_patch(rect)
            
            # Add job info with better formatting
            self.ax_running.text(
                5, y_pos, f'J{job.job_id} | C:{carbon_consideration:.2f}\n{remaining_time:.1f}h left',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=text_color
            )
            
            y_pos -= 1
        
        # Add color legend at bottom
        if running_jobs:
            self.ax_running.text(5, 0.5, '🔴 Low → 🟢 High', 
                               ha='center', va='center', fontsize=8, style='italic')
        
        self.ax_running.set_xlim(0, 10)
        self.ax_running.set_ylim(0, 10)
        self.ax_running.axis('off')
    
    def _plot_statistics(self):
        """Plot episode statistics with carbon emissions data"""
        self.ax_stats.set_title('Statistics & Carbon Emissions', fontweight='bold')
        
        # Calculate basic stats
        total_jobs = self.env.base_env.loads.size()
        finished_jobs_count = 0
        total_emissions = 0
        total_worst_case_emissions = 0
        avg_carbon_consideration = 0
        consideration_count = 0
        
        # Calculate carbon emissions for finished jobs
        for job in self.env.base_env.loads.all_jobs:
            if hasattr(job, 'scheduled_time') and job.scheduled_time != -1:
                # Count carbon consideration for all scheduled jobs
                carbon_consideration = getattr(job, 'carbon_consideration', 0)
                avg_carbon_consideration += carbon_consideration
                consideration_count += 1
                
                # Check if job is finished
                end_time_job = job.scheduled_time + job.run_time
                if end_time_job <= self.current_time:
                    finished_jobs_count += 1
                    actual_emissions, worst_case_emissions = self.calculate_job_carbon_emissions(job)
                    total_emissions += actual_emissions
                    total_worst_case_emissions += worst_case_emissions
        
        # Calculate averages
        if consideration_count > 0:
            avg_carbon_consideration /= consideration_count
        
        emissions_saved = total_worst_case_emissions - total_emissions
        efficiency_pct = (emissions_saved / total_worst_case_emissions * 100) if total_worst_case_emissions > 0 else 0
        
        # Format statistics
        stats_text = f"""🔢 BASIC STATS:
Atomic Step: {self.current_step}
Current Time: {self.current_time/3600:.2f}h
Queue Size: {len(self.env.base_env.job_queue)}
Running Jobs: {len(self.env.base_env.running_jobs)}
Finished Jobs: {finished_jobs_count}/{total_jobs}

🌱 CARBON STATS:
Avg Carbon Consideration: {avg_carbon_consideration:.3f}
Total Actual Emissions: {total_emissions:.0f} gCO2
Total Worst-Case: {total_worst_case_emissions:.0f} gCO2
Emissions Saved: {emissions_saved:.0f} gCO2
Carbon Efficiency: {efficiency_pct:.1f}%

📖 LEGEND:
C: = Carbon Consideration (0-1)
🔴→🟢 = Low to High Consideration
🔵→🔴 = Low to High Emissions"""
        
        self.ax_stats.text(
            0.05, 0.95, stats_text,
            transform=self.ax_stats.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            family='monospace'
        )
        
        self.ax_stats.axis('off')
    
    def _plot_carbon_intensity(self):
        """Plot carbon intensity"""
        self.ax_carbon.set_title('Carbon Intensity (24h window)', fontweight='bold')
        self.ax_carbon.set_xlabel('Simulation Time (hours)')
        self.ax_carbon.set_ylabel('gCO2eq/kWh')
        
        if hasattr(self.env.base_env.cluster, 'carbonIntensity'):
            carbon_data = self.env.base_env.cluster.carbonIntensity.carbonIntensityList
            current_hour_sim = self.current_time / 3600
            
            # Generate 24-hour window
            window = 24
            start_hour_sim = current_hour_sim - window//2
            end_hour_sim = current_hour_sim + window//2
            
            # Create arrays for the window
            sim_hours = np.arange(start_hour_sim, end_hour_sim + 1)
            intensities = []
            
            for hour in sim_hours:
                data_index = int(hour) % len(carbon_data)
                intensities.append(carbon_data[data_index])
            
            # Plot
            self.ax_carbon.plot(sim_hours, intensities, color='green', linewidth=2)
            
            # Current time marker
            self.ax_carbon.axvline(current_hour_sim, color='red', linestyle='--', linewidth=2)
            
            # Set limits
            self.ax_carbon.set_xlim(start_hour_sim, end_hour_sim)
            if intensities:
                margin = (max(intensities) - min(intensities)) * 0.1
                self.ax_carbon.set_ylim(min(intensities) - margin, max(intensities) + margin)
        
        self.ax_carbon.grid(True, alpha=0.3)
    
    def _save_current_state(self):
        """Save complete current environment state"""
        import copy
        
        # Save environment state
        env_state = {
            'current_timestamp': self.env.base_env.current_timestamp,
            'next_arriving_job_idx': self.env.base_env.next_arriving_job_idx,
            'job_queue': copy.deepcopy(self.env.base_env.job_queue),
            'running_jobs': copy.deepcopy(self.env.base_env.running_jobs),
            'scheduled_rl': copy.deepcopy(self.env.base_env.scheduled_rl),
            'current_step': self.current_step,
            'current_time': self.current_time,
            'done': self.done
        }
        
        # Save cluster state including all machine states
        cluster = self.env.base_env.cluster
        cluster_state = {
            'free_node': cluster.free_node,
            'used_node': cluster.used_node,
            'all_nodes': copy.deepcopy(cluster.all_nodes),  # Machine objects with their states
        }
        
        # Save power structure state (this is critical for carbon calculations)
        power_state = {}
        if hasattr(cluster, 'PowerStruc'):
            power_struc = cluster.PowerStruc
            power_state = {
                'currentTime': getattr(power_struc, 'currentTime', 0),
                'jobPowerLogs': copy.deepcopy(getattr(power_struc, 'jobPowerLogs', {})),
                'powerTimeLine': copy.deepcopy(getattr(power_struc, 'powerTimeLine', [])),
            }
        
        # Save carbon intensity state (if it has any internal state)
        carbon_state = {}
        if hasattr(cluster, 'carbonIntensity'):
            carbon_intensity = cluster.carbonIntensity
            carbon_state = {
                'carbonIntensityList': getattr(carbon_intensity, 'carbonIntensityList', []),
                'year': getattr(carbon_intensity, 'year', 2021)
            }
        
        # Save atomic environment state
        atomic_state = {
            'pending_events': copy.deepcopy(self.env.pending_events),
            'current_neural_action': self.env.current_neural_action,
            'episode_done': self.env.episode_done
        }
        
        complete_state = {
            'env_state': env_state,
            'cluster_state': cluster_state,
            'power_state': power_state,
            'carbon_state': carbon_state,
            'atomic_state': atomic_state
        }
        
        self.state_history.append(complete_state)
        
        if self.debug:
            print(f"💾 Saved state #{len(self.state_history)}: {self.current_step} steps, {len(self.env.base_env.job_queue)} queued, {len(self.env.base_env.running_jobs)} running")
    
    def _restore_state(self, state):
        """Restore complete environment state"""
        import copy
        
        # Restore environment state
        env_state = state['env_state']
        self.env.base_env.current_timestamp = env_state['current_timestamp']
        self.env.base_env.next_arriving_job_idx = env_state['next_arriving_job_idx']
        self.env.base_env.job_queue = copy.deepcopy(env_state['job_queue'])
        self.env.base_env.running_jobs = copy.deepcopy(env_state['running_jobs'])
        self.env.base_env.scheduled_rl = copy.deepcopy(env_state['scheduled_rl'])
        
        self.current_step = env_state['current_step']
        self.current_time = env_state['current_time']
        self.done = env_state['done']
        
        # Restore cluster state including all machine states
        cluster = self.env.base_env.cluster
        cluster_state = state['cluster_state']
        cluster.free_node = cluster_state['free_node']
        cluster.used_node = cluster_state['used_node']
        cluster.all_nodes = copy.deepcopy(cluster_state['all_nodes'])
        
        # Restore power structure state
        if 'power_state' in state and hasattr(cluster, 'PowerStruc'):
            power_state = state['power_state']
            power_struc = cluster.PowerStruc
            if 'currentTime' in power_state:
                power_struc.currentTime = power_state['currentTime']
            if 'jobPowerLogs' in power_state:
                power_struc.jobPowerLogs = copy.deepcopy(power_state['jobPowerLogs'])
            if 'powerTimeLine' in power_state:
                power_struc.powerTimeLine = copy.deepcopy(power_state['powerTimeLine'])
        
        # Restore carbon intensity state
        if 'carbon_state' in state and hasattr(cluster, 'carbonIntensity'):
            carbon_state = state['carbon_state']
            carbon_intensity = cluster.carbonIntensity
            if 'carbonIntensityList' in carbon_state:
                carbon_intensity.carbonIntensityList = carbon_state['carbonIntensityList']
            if 'year' in carbon_state:
                carbon_intensity.year = carbon_state['year']
        
        # Restore atomic environment state
        atomic_state = state['atomic_state']
        self.env.pending_events = copy.deepcopy(atomic_state['pending_events'])
        self.env.current_neural_action = atomic_state['current_neural_action']
        self.env.episode_done = atomic_state['episode_done']
        
        # Final synchronization - make sure power structure is updated to current time
        if hasattr(cluster, 'PowerStruc'):
            cluster.PowerStruc.updateCurrentTime(self.current_time)
        
        if self.debug:
            print(f"📁 Restored state: {self.current_step} steps, {len(self.env.base_env.job_queue)} queued, {len(self.env.base_env.running_jobs)} running")
    
    def run(self):
        """Run the interactive visualization"""
        print(f"🚀 Starting Atomic HPC Scheduling Visualization")
        print(f"Model: {self.actor_weights_path}")
        print(f"Workload: {self.workload}")
        print(f"Backfill: {self.backfill}")
        print(f"\n🎮 Controls:")
        print(f"  - Reset: Start new episode")
        print(f"  - Next Event: Take next atomic step")
        print(f"  - Neural Step: Take only neural decisions")
        print(f"  - Back: Step back one event")
        
        plt.show()

def find_model_path(training_dir, epoch=None):
    """
    Find the model path from training directory and epoch
    
    Args:
        training_dir: Path to training directory (e.g., 'lublin_256_carbon_float_simple/MARL_direct_co2_no_wait_no_backfill_incread_lr')
        epoch: Epoch number (optional, defaults to 'final' if not specified)
    
    Returns:
        Path to the model file
    """
    if epoch is None:
        # Try final directory first
        final_path = os.path.join(training_dir, 'final', '_actor.pkl')
        if os.path.exists(final_path):
            return final_path
        
        # Fall back to root directory
        root_path = os.path.join(training_dir, '_actor.pkl')
        if os.path.exists(root_path):
            return root_path
        
        raise FileNotFoundError(f"No model found in {training_dir}/final/ or {training_dir}/")
    else:
        # Try checkpoints directory
        checkpoint_path = os.path.join(training_dir, 'checkpoints', f'epoch_{epoch}', '_actor.pkl')
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        
        raise FileNotFoundError(f"No model found at {checkpoint_path}")

def extract_workload_from_path(training_dir):
    """Extract workload name from training directory path"""
    # Split by path separator and take the first part which should be the workload
    parts = training_dir.split(os.path.sep)
    if len(parts) > 0:
        workload_part = parts[0]
        # Remove any suffix like '_simple' but keep base workload name
        if workload_part.startswith('lublin_256_carbon'):
            return 'lublin_256_carbon'
        else:
            return workload_part
    return 'lublin_256_carbon'  # Default fallback

def main():
    parser = argparse.ArgumentParser(
        description='Atomic HPC Scheduling Visualizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use final model from training directory
  python visualize_atomic_scheduling.py lublin_256_carbon_float_simple/MARL_direct_co2_no_wait_no_backfill_incread_lr
  
  # Use specific epoch
  python visualize_atomic_scheduling.py lublin_256_carbon_float_simple/MARL_direct_co2_no_wait_no_backfill_incread_lr --epoch 20
  
  # Use direct model path (legacy)
  python visualize_atomic_scheduling.py --model path/to/_actor.pkl
        """
    )
    
    # Primary argument: training directory OR --model path
    parser.add_argument('training_dir', nargs='?', help='Training directory path (e.g., lublin_256_carbon_float_simple/MARL_direct_co2_no_wait_no_backfill_incread_lr)')
    parser.add_argument('--model', help='Direct path to trained model weights (alternative to training_dir)')
    parser.add_argument('--epoch', type=int, help='Epoch number (uses final model if not specified)')
    parser.add_argument('--workload', help='Workload name (auto-detected from training_dir if not specified)')
    parser.add_argument('--backfill', type=int, default=3, help='Backfill strategy')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Determine model path and workload
    if args.model:
        # Legacy mode: direct model path
        model_path = args.model
        workload = args.workload or 'lublin_256_carbon'
        print(f"📁 Using direct model path: {model_path}")
    elif args.training_dir:
        # New mode: training directory + epoch
        try:
            model_path = find_model_path(args.training_dir, args.epoch)
            workload = args.workload or extract_workload_from_path(args.training_dir)
            
            epoch_str = f"epoch {args.epoch}" if args.epoch else "final"
            print(f"📁 Found model: {model_path} ({epoch_str})")
            print(f"🎯 Detected workload: {workload}")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print(f"\n💡 Available options in {args.training_dir}:")
            
            # Show available epochs
            checkpoints_dir = os.path.join(args.training_dir, 'checkpoints')
            if os.path.exists(checkpoints_dir):
                epochs = []
                for item in os.listdir(checkpoints_dir):
                    if item.startswith('epoch_') and os.path.isdir(os.path.join(checkpoints_dir, item)):
                        epoch_num = item.replace('epoch_', '')
                        if os.path.exists(os.path.join(checkpoints_dir, item, '_actor.pkl')):
                            epochs.append(int(epoch_num))
                
                if epochs:
                    epochs.sort()
                    print(f"   Available epochs: {epochs}")
                else:
                    print(f"   No epoch checkpoints found")
            
            # Show final model
            final_path = os.path.join(args.training_dir, 'final', '_actor.pkl')
            if os.path.exists(final_path):
                print(f"   Final model: available")
            else:
                print(f"   Final model: not found")
            
            return
    else:
        parser.error("Must specify either training_dir or --model")
    
    # Create and run visualizer
    try:
        visualizer = AtomicSchedulingVisualizer(
            actor_weights_path=model_path,
            workload=workload,
            backfill=args.backfill,
            debug=args.debug
        )
        
        print(f"🎮 Starting visualization...")
        print(f"   Model: {model_path}")
        print(f"   Workload: {workload}")
        print(f"   Backfill: {args.backfill}")
        
        visualizer.run()
        
    except Exception as e:
        print(f"❌ Error creating visualizer: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 