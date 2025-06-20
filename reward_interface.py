""" #!/usr/bin/env python3
"""
Reward Function Interface System for HPC Job Scheduling

This module provides a flexible interface for implementing and testing different
reward functions in the HPC scheduling environment. It includes:

1. Abstract base class for reward functions
2. Concrete implementations of different reward strategies
3. Factory pattern for instantiating reward functions
4. Configuration-based reward function selection
"""

from abc import ABC, abstractmethod
import configparser
import importlib
import inspect
from typing import List, Dict, Any, Optional
import numpy as np


class RewardFunction(ABC):
    """
    Abstract base class for all reward functions.
    
    Defines the interface that all reward implementations must follow.
    """
    
    def __init__(self, config: configparser.ConfigParser, **kwargs):
        """
        Initialize the reward function with configuration.
        
        Args:
            config: Configuration parser object
            **kwargs: Additional parameters specific to the reward function
        """
        self.config = config
        self.kwargs = kwargs
        self.setup()
    
    @abstractmethod
    def setup(self):
        """Setup method called during initialization. Override for custom setup."""
        pass
    
    @abstractmethod
    def calculate_reward(self, scheduled_jobs: List, context: Dict[str, Any] = None) -> float:
        """
        Calculate the reward for a set of scheduled jobs.
        
        Args:
            scheduled_jobs: List of job objects with scheduling information
            context: Additional context information (environment state, etc.)
            
        Returns:
            Reward value (typically negative for penalties, positive for rewards)
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this reward function."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this reward function."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return the parameters used by this reward function."""
        return self.kwargs
    
    def validate_jobs(self, scheduled_jobs: List) -> List:
        """
        Validate and filter scheduled jobs.
        
        Args:
            scheduled_jobs: List of job objects
            
        Returns:
            List of valid scheduled jobs
        """
        valid_jobs = []
        for job in scheduled_jobs:
            if hasattr(job, 'scheduled_time') and job.scheduled_time != -1:
                valid_jobs.append(job)
        return valid_jobs


class CarbonAwareReward(RewardFunction):
    """
    Carbon-aware reward function that matches the original MARL implementation.
    This should produce exactly the same results as the original greenRwd calculation.
    """
    
    def setup(self):
        """Setup carbon intensity calculator."""
        # Import here to avoid circular imports
        from greenPower import carbon_intensity
        
        # Get parameters from config
        green_win = int(self.config.get('GAS-MARL setting', 'green_win', fallback=24))
        carbon_year = int(self.config.get('general setting', 'carbon_year', fallback=2021))
        
        # Initialize carbon intensity calculator
        self.carbon_intensity = carbon_intensity(green_win, carbon_year)
        
        # Set start offset if provided in context
        if 'start_offset' in self.kwargs:
            self.carbon_intensity.setStartOffset(self.kwargs['start_offset'])
    
    def calculate_reward(self, scheduled_jobs: List, context: Dict[str, Any] = None) -> float:
        """Calculate carbon-aware reward using the original algorithm."""
        # Handle special case where we're passed environment context
        if context and 'env' in context:
            env = context['env']
            try:
                # Get all scheduled jobs from the environment
                scheduledJobs = []
                
                # In the original implementation, this happens after the job is scheduled
                if hasattr(env, 'loads') and hasattr(env.loads, 'all_jobs'):
                    scheduledJobs = [j for j in env.loads.all_jobs 
                                   if hasattr(j, 'scheduled_time') and j.scheduled_time != -1]
                
                # Call the original carbon reward calculation
                if hasattr(env, 'cluster') and hasattr(env.cluster, 'carbonIntensity'):
                    carbon_reward = env.cluster.carbonIntensity.getCarbonAwareReward(scheduledJobs)
                    return carbon_reward
                else:
                    # Fallback if environment doesn't have the expected structure
                    return 0.0
                    
            except Exception as e:
                print(f"Warning: Error calculating carbon reward: {e}")
                return 0.0
        else:
            # Standard interface - use our own carbon intensity calculator
            return self.carbon_intensity.getCarbonAwareReward(scheduled_jobs)
    
    def get_name(self) -> str:
        return "CarbonAware"
    
    def get_description(self) -> str:
        return "Original carbon-aware reward based on emission ratios weighted by carbon consideration"


class SimpleEmissionReward(RewardFunction):
    """
    Simple emission-based reward function.
    
    Directly penalizes total carbon emissions without complex ratio calculations.
    """
    
    def setup(self):
        """Setup carbon intensity calculator."""
        from greenPower import carbon_intensity
        
        green_win = int(self.config.get('GAS-MARL setting', 'green_win', fallback=200))
        carbon_year = int(self.config.get('general setting', 'carbon_year', fallback=2021))
        
        self.carbon_intensity = carbon_intensity(green_win, carbon_year)
        self.emission_weight = self.kwargs.get('emission_weight', 1.0)
        
        if 'start_offset' in self.kwargs:
            self.carbon_intensity.setStartOffset(self.kwargs['start_offset'])
    
    def calculate_reward(self, scheduled_jobs: List, context: Dict[str, Any] = None) -> float:
        """Calculate reward based on total carbon emissions."""
        valid_jobs = self.validate_jobs(scheduled_jobs)
        
        if not valid_jobs:
            return 0.0
        
        total_emissions = 0.0
        total_weight = 0.0
        
        for job in valid_jobs:
            start_time = job.scheduled_time
            end_time = start_time + job.request_time
            power = job.power
            carbon_consideration = getattr(job, 'carbon_consideration', 1.0)
            
            # Calculate emissions for this job
            emissions = self.carbon_intensity.getCarbonEmissions(power, start_time, end_time)
            
            # Weight by carbon consideration
            weighted_emissions = emissions * carbon_consideration
            total_emissions += weighted_emissions
            total_weight += carbon_consideration
        
        # Normalize by total weight and apply emission weight
        if total_weight > 0:
            avg_weighted_emissions = total_emissions / total_weight
            return -avg_weighted_emissions * self.emission_weight
        
        return 0.0
    
    def get_name(self) -> str:
        return "SimpleEmission"
    
    def get_description(self) -> str:
        return f"Simple emission-based reward with weight {self.emission_weight}"


class HybridReward(RewardFunction):
    """
    Hybrid reward function combining multiple objectives.
    
    Combines carbon emissions, wait times, and utilization metrics.
    """
    
    def setup(self):
        """Setup hybrid reward components."""
        from greenPower import carbon_intensity
        
        green_win = int(self.config.get('GAS-MARL setting', 'green_win', fallback=200))
        carbon_year = int(self.config.get('general setting', 'carbon_year', fallback=2021))
        
        self.carbon_intensity = carbon_intensity(green_win, carbon_year)
        
        # Weights for different components
        self.carbon_weight = self.kwargs.get('carbon_weight', 0.5)
        self.wait_weight = self.kwargs.get('wait_weight', 0.3)
        self.utilization_weight = self.kwargs.get('utilization_weight', 0.2)
        
        # Normalization factors
        self.max_wait_time = float(self.config.get('algorithm constants', 'MAX_WAIT_TIME', fallback=43200))
        
        if 'start_offset' in self.kwargs:
            self.carbon_intensity.setStartOffset(self.kwargs['start_offset'])
    
    def calculate_reward(self, scheduled_jobs: List, context: Dict[str, Any] = None) -> float:
        """Calculate hybrid reward combining multiple objectives."""
        valid_jobs = self.validate_jobs(scheduled_jobs)
        
        if not valid_jobs:
            return 0.0
        
        # Carbon component
        carbon_reward = 0.0
        if self.carbon_weight > 0:
            carbon_reward = self.carbon_intensity.getCarbonAwareReward(valid_jobs)
        
        # Wait time component
        wait_reward = 0.0
        if self.wait_weight > 0:
            total_wait = 0.0
            for job in valid_jobs:
                wait_time = job.scheduled_time - job.submit_time
                normalized_wait = min(wait_time / self.max_wait_time, 1.0)
                total_wait += normalized_wait
            
            avg_wait = total_wait / len(valid_jobs) if valid_jobs else 0.0
            wait_reward = -avg_wait  # Negative because longer waits are bad
        
        # Utilization component (simplified)
        utilization_reward = 0.0
        if self.utilization_weight > 0 and context:
            # Use makespan or other efficiency metrics if available in context
            makespan = context.get('makespan', 0)
            if makespan > 0:
                # Simple utilization metric (lower makespan = better)
                utilization_reward = -makespan / 86400.0  # Normalize by day
        
        # Combine components
        total_reward = (self.carbon_weight * carbon_reward + 
                       self.wait_weight * wait_reward + 
                       self.utilization_weight * utilization_reward)
        
        return total_reward
    
    def get_name(self) -> str:
        return "Hybrid"
    
    def get_description(self) -> str:
        return (f"Hybrid reward: carbon({self.carbon_weight}) + "
                f"wait({self.wait_weight}) + utilization({self.utilization_weight})")


class DelayPenaltyReward(RewardFunction):
    """
    Delay-penalty based reward function.
    
    Focuses on minimizing job delays relative to their priorities.
    """
    
    def setup(self):
        """Setup delay penalty parameters."""
        self.delay_weight = self.kwargs.get('delay_weight', 1.0)
        self.priority_weight = self.kwargs.get('priority_weight', 0.5)
        self.max_wait_time = float(self.config.get('algorithm constants', 'MAX_WAIT_TIME', fallback=43200))
    
    def calculate_reward(self, scheduled_jobs: List, context: Dict[str, Any] = None) -> float:
        """Calculate reward based on job delays and priorities."""
        valid_jobs = self.validate_jobs(scheduled_jobs)
        
        if not valid_jobs:
            return 0.0
        
        total_penalty = 0.0
        
        for job in valid_jobs:
            # Calculate delay
            delay = job.scheduled_time - job.submit_time
            normalized_delay = min(delay / self.max_wait_time, 1.0)
            
            # Get job priority (use carbon_consideration as proxy for priority)
            priority = getattr(job, 'carbon_consideration', 0.5)
            
            # Calculate penalty (higher priority jobs get higher penalty for delays)
            priority_factor = 1.0 + (priority * self.priority_weight)
            job_penalty = normalized_delay * priority_factor
            
            total_penalty += job_penalty
        
        # Return negative average penalty
        avg_penalty = total_penalty / len(valid_jobs) if valid_jobs else 0.0
        return -avg_penalty * self.delay_weight
    
    def get_name(self) -> str:
        return "DelayPenalty"
    
    def get_description(self) -> str:
        return f"Delay penalty reward with delay_weight={self.delay_weight}, priority_weight={self.priority_weight}"


class RewardFunctionFactory:
    """
    Factory class for creating reward function instances.
    
    Supports both built-in reward functions and custom implementations.
    """
    
    # Registry of built-in reward functions
    BUILTIN_REWARDS = {
        'carbon_aware': CarbonAwareReward,
        'simple_emission': SimpleEmissionReward,
        'hybrid': HybridReward,
        'delay_penalty': DelayPenaltyReward,
    }
    
    @classmethod
    def create_reward_function(cls, reward_type: str, config: configparser.ConfigParser, **kwargs) -> RewardFunction:
        """
        Create a reward function instance.
        
        Args:
            reward_type: Type of reward function to create
            config: Configuration parser object
            **kwargs: Additional parameters for the reward function
            
        Returns:
            RewardFunction instance
            
        Raises:
            ValueError: If reward_type is not recognized
        """
        # Check built-in rewards first
        if reward_type.lower() in cls.BUILTIN_REWARDS:
            reward_class = cls.BUILTIN_REWARDS[reward_type.lower()]
            return reward_class(config, **kwargs)
        
        # Try to import custom reward function
        try:
            # Assume format: "module.ClassName"
            if '.' in reward_type:
                module_name, class_name = reward_type.rsplit('.', 1)
                module = importlib.import_module(module_name)
                reward_class = getattr(module, class_name)
            else:
                # Try to import from current module
                reward_class = globals().get(reward_type)
                if reward_class is None:
                    raise ImportError(f"Reward function {reward_type} not found")
            
            # Verify it's a proper reward function
            if not (inspect.isclass(reward_class) and issubclass(reward_class, RewardFunction)):
                raise TypeError(f"{reward_type} is not a valid RewardFunction subclass")
            
            return reward_class(config, **kwargs)
            
        except (ImportError, AttributeError, TypeError) as e:
            raise ValueError(f"Cannot create reward function '{reward_type}': {e}")
    
    @classmethod
    def list_available_rewards(cls) -> List[str]:
        """Return list of available built-in reward functions."""
        return list(cls.BUILTIN_REWARDS.keys())
    
    @classmethod
    def get_reward_info(cls, reward_type: str) -> Dict[str, str]:
        """Get information about a specific reward function."""
        if reward_type.lower() in cls.BUILTIN_REWARDS:
            reward_class = cls.BUILTIN_REWARDS[reward_type.lower()]
            # Create temporary instance to get info
            dummy_config = configparser.ConfigParser()
            instance = reward_class(dummy_config)
            return {
                'name': instance.get_name(),
                'description': instance.get_description(),
                'class': reward_class.__name__
            }
        else:
            return {'name': reward_type, 'description': 'Custom reward function', 'class': 'Unknown'}


class RewardManager:
    """
    Manager class for handling reward function configuration and usage.
    
    Provides a high-level interface for using different reward functions.
    """
    
    def __init__(self, config_file: str = 'configFile/config.ini'):
        """
        Initialize the reward manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # Load reward configuration
        self.reward_type = self.config.get('reward_config', 'reward_function', fallback='carbon_aware')
        self.reward_params = self._load_reward_params()
        
        # Create reward function instance
        self.reward_function = RewardFunctionFactory.create_reward_function(
            self.reward_type, self.config, **self.reward_params
        )
    
    def _load_reward_params(self) -> Dict[str, Any]:
        """Load reward function parameters from config."""
        params = {}
        
        # Check if there's a reward-specific section
        section_name = f'reward_{self.reward_type}'
        if self.config.has_section(section_name):
            for key, value in self.config.items(section_name):
                # Try to convert to appropriate type
                try:
                    # Try float first
                    params[key] = float(value)
                except ValueError:
                    try:
                        # Try int
                        params[key] = int(value)
                    except ValueError:
                        # Keep as string
                        params[key] = value
        
        return params
    
    def calculate_reward(self, scheduled_jobs: List, context: Dict[str, Any] = None) -> float:
        """
        Calculate reward using the configured reward function.
        
        Args:
            scheduled_jobs: List of scheduled job objects
            context: Additional context information
            
        Returns:
            Calculated reward value
        """
        return self.reward_function.calculate_reward(scheduled_jobs, context)
    
    def calculate_episode_rewards(self, context: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Calculate all reward components for a completed episode.
        
        This method calculates all major reward components:
        - Carbon-aware reward (green reward) 
        - Wait time penalty
        - Total combined reward using eta weighting
        
        Args:
            context: Environment context including completed episode data
            
        Returns:
            Dictionary containing all reward components:
            {
                'carbon_reward': float,     # Carbon-aware component
                'wait_reward': float,       # Wait time penalty  
                'total_reward': float,      # Combined reward (eta * wait + carbon)
                'specialized_reward': float # Reserved for future specialized rewards
            }
        """
        if not context or 'env' not in context:
            return {
                'carbon_reward': 0.0,
                'wait_reward': 0.0,
                'total_reward': 0.0,
                'specialized_reward': 0.0
            }
        
        env = context['env']
        rewards = {}
        
        # 1. Calculate carbon-aware reward (green reward)
        rewards['carbon_reward'] = self._calculate_carbon_reward(env)
        
        # 2. Calculate wait time penalty
        rewards['wait_reward'] = self._calculate_wait_reward(env)
        
        # 3. Calculate specialized reward if applicable
        rewards['specialized_reward'] = 0.0
            
        # 4. Calculate total combined reward using eta weighting
        eta = float(self.config.get('GAS-MARL setting', 'eta', fallback=0.5))
        rewards['total_reward'] = eta * rewards['wait_reward'] + rewards['carbon_reward']
        
        return rewards
    
    def _calculate_carbon_reward(self, env) -> float:
        """
        Calculate carbon-aware reward for the completed episode.
        
        Uses the original greenPower.getCarbonAwareReward() function
        but calculated at episode completion rather than incrementally.
        """
        try:
            # Get all scheduled jobs from the completed episode
            if hasattr(env, 'loads') and hasattr(env.loads, 'all_jobs'):
                scheduled_jobs = [j for j in env.loads.all_jobs 
                                if hasattr(j, 'scheduled_time') and j.scheduled_time != -1]
                
                if scheduled_jobs:
                    # Use environment's carbon intensity instance to calculate reward
                    if hasattr(env, 'cluster') and hasattr(env.cluster, 'carbonIntensity'):
                        carbon_reward = env.cluster.carbonIntensity.getCarbonAwareReward(scheduled_jobs)
                        return carbon_reward
                    else:
                        # Fallback: create our own carbon intensity calculator
                        from greenPower import carbon_intensity
                        green_win = int(self.config.get('GAS-MARL setting', 'green_win', fallback=200))
                        carbon_year = int(self.config.get('general setting', 'carbon_year', fallback=2021))
                        carbon_calc = carbon_intensity(green_win, carbon_year)
                        carbon_reward = carbon_calc.getCarbonAwareReward(scheduled_jobs)
                        return carbon_reward
            
            return 0.0
            
        except Exception as e:
            print(f"Warning: Error calculating carbon reward: {e}")
            return 0.0
    
    def _calculate_wait_reward(self, env) -> float:
        """
        Calculate cumulative wait time penalty for the completed episode.
        
        This replicates the wait time penalty calculation that would normally
        happen incrementally during the episode, but calculated all at once
        at episode completion.
        """
        try:
            if hasattr(env, 'loads') and hasattr(env.loads, 'all_jobs'):
                total_wait_penalty = 0.0
                processed_jobs = 0
                
                for job in env.loads.all_jobs:
                    if hasattr(job, 'scheduled_time') and job.scheduled_time != -1:
                        # Calculate wait time penalty for this job
                        if hasattr(job, 'submit_time'):
                            wait_time = job.scheduled_time - job.submit_time
                            if wait_time > 0:
                                # Use negative wait time as penalty (similar to original step-by-step calculation)
                                # This needs to match the original reward calculation logic
                                job_penalty = -wait_time / 3600.0  # Convert to hours and make negative
                                total_wait_penalty += job_penalty
                                processed_jobs += 1
                
                # Return total cumulative wait penalty
                return total_wait_penalty
            
            return 0.0
            
        except Exception as e:
            print(f"Warning: Error calculating wait reward: {e}")
            return 0.0
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get information about the current reward function."""
        return {
            'type': self.reward_type,
            'name': self.reward_function.get_name(),
            'description': self.reward_function.get_description(),
            'parameters': self.reward_function.get_parameters()
        }
    
    def set_start_offset(self, offset: int):
        """Set start offset for carbon intensity data (if applicable)."""
        if hasattr(self.reward_function, 'carbon_intensity'):
            self.reward_function.carbon_intensity.setStartOffset(offset)


# Convenience function for backward compatibility
def get_carbon_aware_reward(scheduled_jobs: List, config_file: str = 'configFile/config.ini') -> float:
    """
    Convenience function to get carbon-aware reward (backward compatibility).
    
    Args:
        scheduled_jobs: List of scheduled job objects
        config_file: Path to configuration file
        
    Returns:
        Carbon-aware reward value
    """
    manager = RewardManager(config_file)
    return manager.calculate_reward(scheduled_jobs)


if __name__ == "__main__":
    # Example usage and testing
    print("Available reward functions:")
    for reward_type in RewardFunctionFactory.list_available_rewards():
        info = RewardFunctionFactory.get_reward_info(reward_type)
        print(f"  {reward_type}: {info['description']}")
    
    # Example configuration
    print("\nExample reward manager usage:")
    manager = RewardManager()
    print(f"Current reward function: {manager.get_reward_info()}")  """