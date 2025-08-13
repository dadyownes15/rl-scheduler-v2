import numpy as np
import pandas as pd
import os


class CarbonSampler:
    """
    Carbon intensity sampler for curriculum learning.
    Controls the difficulty of carbon data by interpolating between:
    - tau=0: Easiest (flat mean carbon intensity)
    - tau=1: Hardest (real historical trace with full variance)
    """
    
    def __init__(self, hist, horizon=24):
        """
        Initialize the carbon sampler.
        
        Args:
            hist: numpy array of historical carbon intensity values
            horizon: number of hours to sample (default 24)
        """
        self.hist = hist
        self.H = horizon
        self.mu = hist.mean()
        self.sigma = hist.std()
        
    @classmethod
    def from_csv(cls, csv_path="data/DK-DK2_hourly_carbon_intensity_noFeb29.csv", 
                 year=2021, horizon=24):
        """
        Create CarbonSampler from CSV file.
        
        Args:
            csv_path: Path to carbon intensity CSV file
            year: Which year column to use (2021, 2022, 2023, 2024)
            horizon: Number of hours to sample
            
        Returns:
            CarbonSampler instance
        """
        # Map year to column index
        year_to_col = {2021: 1, 2022: 2, 2023: 3, 2024: 4}
        col_index = year_to_col.get(year, 1)  # Default to 2021
        
        # Load carbon intensity data
        current_dir = os.getcwd()
        full_path = os.path.join(current_dir, csv_path)
        
        carbon_list = []
        with open(full_path, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                row = line.strip().split(',')
                if len(row) > col_index:
                    carbon_list.append(float(row[col_index]))
        
        hist = np.array(carbon_list)
        return cls(hist, horizon)
    
    def sample(self, tau: float) -> np.ndarray:
        """
        Sample carbon intensity profile based on curriculum parameter tau.
        Uses random walk for temporal correlation instead of i.i.d sampling.
        
        Args:
            tau: Curriculum parameter [0, 1]
                 0 = easiest (flat mean)
                 1 = hardest (real historical trace)
                 
        Returns:
            numpy array of carbon intensities for H hours
        """
        tau = np.clip(tau, 0., 1.)
        
        if tau == 1.0:
            # Hardest: real historical trace
            i = np.random.randint(0, len(self.hist) - self.H)
            return self.hist[i:i + self.H]
        
        if tau == 0.0:
            # Easiest: flat mean
            return np.full(self.H, self.mu)
        
        # Intermediate: random walk starting from mean
        # The walk step size scales with tau (more variance = larger steps)
        profile = np.zeros(self.H)
        profile[0] = self.mu  # Start at mean
        
        # Random walk parameters
        step_scale = tau * self.sigma * 0.2  # Scale step size with tau
        drift_back = 0.05  # Tendency to drift back toward mean
        
        for t in range(1, self.H):
            # Random walk step
            step = np.random.normal(0, step_scale)
            
            # Drift back toward mean (prevents excessive wandering)
            drift = -drift_back * (profile[t-1] - self.mu)
            
            # Update with step and drift
            profile[t] = profile[t-1] + step + drift
        
        # Ensure non-negative values
        return np.clip(profile, 0, None)
    
    def get_stats(self):
        """
        Get statistics about the historical carbon intensity data.
        
        Returns:
            Dictionary with mean, std, min, max of historical data
        """
        return {
            'mean': self.mu,
            'std': self.sigma,
            'min': self.hist.min(),
            'max': self.hist.max(),
            'length': len(self.hist)
        } 