import csv
import os
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('configFile/config.ini')

# Carbon window configuration
USE_DYNAMIC_WINDOW = config.getboolean('carbon setting', 'USE_DYNAMIC_WINDOW', fallback=True)  # If True, use dynamic window from scheduled time to last job end
                          # If False, use fixed 24-hour window from scheduled time

# Carbon reward function configuration
CARBON_REWARD_FUNCTION = config.get('carbon setting', 'carbon_reward_function', fallback='emission_ratio')

# Maximum carbon intensity for normalization (gCO2eq/kWh)
# Based on the historical maximum from DK data: ~467.25, rounded up to 500 for safety
MAX_CARBON_INTENSITY = float(config.get('algorithm constants', 'MAX_CARBON_INTENSITY')) 

class carbon_intensity():
    def __init__(self, greenWin, year=2021):
        """
        Initialize carbon intensity class
        greenWin: time window for carbon intensity data
        year: which year column to use from the CSV (2021, 2022, 2023, 2024)
        """
        self.greenWin = greenWin
        self.year = year
        self.carbonIntensityList = self.loadCarbonIntensityData()
        # How many hours we shift the carbon-intensity timeline (set each episode)
        self.start_offset = 0
    
    # -------------------------------------------------
    #  Helpers for the environment to control the offset
    # -------------------------------------------------
    def setStartOffset(self, offset_hours: int):
        """Shift the carbon-intensity timeline by <offset_hours> (0-8759)."""
        self.start_offset = offset_hours % len(self.carbonIntensityList)
    
    def loadCarbonIntensityData(self):
        """Load carbon intensity data from CSV file"""
        current_dir = os.getcwd()
        carbon_file = os.path.join(current_dir, "./data/DK-DK2_hourly_carbon_intensity_noFeb29.csv")
        
        # Map year to column index
        year_to_col = {2021: 1, 2022: 2, 2023: 3, 2024: 4}
        col_index = year_to_col.get(self.year, 1)  # Default to 2021
        
        carbon_list = []
        with open(carbon_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                carbon_list.append(float(row[col_index]))
        
        return carbon_list
    
    def getCarbonIntensitySlot(self, currentTime):
        """
        Get carbon intensity slots for the next greenWin hours
        Returns list of dicts with 'lastTime' and 'carbonIntensity' (gCO2eq/kWh)
        """
        carbonSlot = []
        index = int(currentTime / 3600)  # Current hour index
        t = currentTime
        
        for i in range(index, index + self.greenWin):
            # Handle wrap-around for year-long data with start offset
            hour_index = (i + self.start_offset) % len(self.carbonIntensityList)
            carbonIntensity = self.carbonIntensityList[hour_index]
            lastTime = (i + 1) * 3600 - t
            carbonSlot.append({'lastTime': lastTime, 'carbonIntensity': carbonIntensity})
            t = (i + 1) * 3600
        
        return carbonSlot
    
    def getCarbonEmissions(self, power, start, end):
        """
        Calculate total carbon emissions for a given power consumption over time period
        power: power consumption in watts
        start, end: time period in seconds
        Returns: total carbon emissions in gCO2eq
        """
        totalEmissions = 0
        startIndex = int(start / 3600)
        endIndex = int(end / 3600)
        t = start
        
        for i in range(startIndex, endIndex + 1):
            if i == endIndex:
                lastTime = end - t
            else:
                lastTime = (i + 1) * 3600 - t
            
            # Handle wrap-around for year-long data with start offset
            hour_index = (i + self.start_offset) % len(self.carbonIntensityList)
            carbonIntensity = self.carbonIntensityList[hour_index]
            
            # Convert power from watts to kW and time from seconds to hours
            energyKWh = (power / 1000.0) * (lastTime / 3600.0)
            emissions = energyKWh * carbonIntensity  # gCO2eq
            totalEmissions += emissions
            
            t = (i + 1) * 3600
        
        return totalEmissions
    
    # Simple wrapper used by Cluster.backfill_check(..)
    def getCarbonEmissionsEstimate(self, power, start, end, *_):
        return self.getCarbonEmissions(power, start, end)

    def getWorstCarbonIntensitiesInPeriod(self, start_time, end_time, job_length_seconds):
        """
        Get the list of carbon intensities that would be used in worst-case calculation,
        accounting for fractional hours.
        
        Args:
            start_time: start of the time window in seconds
            end_time: end of the time window in seconds  
            job_length_seconds: duration of the job in seconds
            
        Returns:
            List of (intensity, hours_used) tuples showing which intensities are used and for how long
        """
        # Input validation
        assert start_time >= 0, "start_time must be non-negative"
        assert end_time > start_time, "end_time must be greater than start_time"
        assert job_length_seconds > 0, "job_length_seconds must be positive"
        
        window_duration = end_time - start_time
        assert window_duration >= job_length_seconds, \
            f"Time window ({window_duration}s) must be at least as long as job length ({job_length_seconds}s)"
            
        # Convert job length to hours (exact, no ceiling)
        job_length_hours = job_length_seconds / 3600.0
        
        # Get all carbon intensities in the time window
        start_hour = int(start_time / 3600)
        end_hour = int(end_time / 3600)
        
        intensities = []
        for hour in range(start_hour, end_hour + 1):
            # Apply start offset and wrap around year
            hour_index = (hour + self.start_offset) % len(self.carbonIntensityList)
            intensity = self.carbonIntensityList[hour_index]
            intensities.append(intensity)
            
        # Sort intensities in descending order (worst first)
        intensities.sort(reverse=True)
        
        # Calculate which intensities are used and for how long
        used_intensities = []
        remaining_hours = job_length_hours
        
        for intensity in intensities:
            if remaining_hours <= 0:
                break
                
            # Take full hour or remaining fraction
            hours_to_use = min(1.0, remaining_hours)
            used_intensities.append((intensity, hours_to_use))
            remaining_hours -= hours_to_use
        
        # If we have a partial hour remaining, use the least intensity available
        if remaining_hours > 0:
            # Sort intensities in ascending order to get the least
            intensities_asc = sorted(intensities)
            if intensities_asc:
                least_intensity = intensities_asc[0]
                used_intensities.append((least_intensity, remaining_hours))
        
        return used_intensities

    def getMaxCarbonIntensityInPeriod(self, start_time, end_time, job_length_seconds):
        """
        Find the worst carbon intensities within a time window for a job of given length,
        accounting for fractional hours to remove ceiling bias.
        
        For partial hours, the least carbon intensity is used (best case for that fraction).
        
        Args:
            start_time: start of the time window in seconds
            end_time: end of the time window in seconds  
            job_length_seconds: duration of the job in seconds
            
        Returns:
            Weighted sum of carbon intensities (gCO2eq/kWh) accounting for fractional hours
        """
        # Input validation
        assert start_time >= 0, "start_time must be non-negative"
        assert end_time > start_time, "end_time must be greater than start_time"
        assert job_length_seconds > 0, "job_length_seconds must be positive"
        
            
        # Convert job length to hours (exact, no ceiling)
        job_length_hours = job_length_seconds / 3600.0
        
        # Get all carbon intensities in the time window
        start_hour = int(start_time / 3600)
        end_hour = int(end_time / 3600)
        
        intensities = []
        for hour in range(start_hour, end_hour + 1):
            # Apply start offset and wrap around year
            hour_index = (hour + self.start_offset) % len(self.carbonIntensityList)
            intensity = self.carbonIntensityList[hour_index]
            intensities.append(intensity)
            
        # Sort intensities in descending order (worst first)
        intensities.sort(reverse=True)
        
        # Calculate weighted sum accounting for fractional hours
        total_weighted_intensity = 0.0
        remaining_hours = job_length_hours
        
        for intensity in intensities:
            if remaining_hours <= 0:
                break
                
            # Take full hour or remaining fraction
            hours_to_use = min(1.0, remaining_hours)
            total_weighted_intensity += intensity * hours_to_use
            remaining_hours -= hours_to_use
        
        # If we have a partial hour remaining, use the least intensity available
        # (best case for the partial hour to reduce bias)
        if remaining_hours > 0:
            # Sort intensities in ascending order to get the least
            intensities_asc = sorted(intensities)
            if intensities_asc:
                least_intensity = intensities_asc[0]
                total_weighted_intensity += least_intensity * remaining_hours
        
        return total_weighted_intensity

    def getMaxCarbonIntensityFromJobs(self, scheduled_jobs):
        """
        Get the worst-case carbon intensities for all scheduled jobs.
        Window configuration controlled by USE_DYNAMIC_WINDOW:
        - If True: dynamic window from each job's scheduled time to when last job ends
        - If False: fixed 24-hour window from each job's scheduled time
        
        Args:
            scheduled_jobs: list of jobs with submit_time, scheduled_time, and request_time attributes
            
        Returns:
            Dictionary with job-wise worst-case carbon intensity sums
        """
        if not scheduled_jobs:
            return {}
            
        # Filter out unscheduled jobs
        valid_jobs = [job for job in scheduled_jobs 
                     if hasattr(job, 'scheduled_time') and job.scheduled_time != -1]
        
        if not valid_jobs:
            return {}
        
        # Determine window end time based on configuration
        if USE_DYNAMIC_WINDOW:
            # Dynamic window: find when the last job ends
            last_job_end = max(job.scheduled_time + job.request_time for job in valid_jobs)

        else:
            # Fixed window will be calculated per job (24 hours from each job's start)
            pass
            
        # Calculate worst-case carbon intensity for each job
        results = {}
        for i, job in enumerate(valid_jobs):
            job_id = getattr(job, 'job_id', f'job_{i}')
            
            # Window configuration
            window_start = job.scheduled_time
            if USE_DYNAMIC_WINDOW:
                # Dynamic: window from job start to when last job ends
                window_end = last_job_end
            else:
                # Fixed: 24-hour window from job start
                window_end = job.scheduled_time + (24 * 3600)  # 24 hours in seconds
            
            # Ensure window is at least as long as the job
            min_window_end = job.scheduled_time + job.request_time
            window_end = max(window_end, min_window_end)
            
            worst_case = self.getMaxCarbonIntensityInPeriod(
                window_start, 
                window_end, 
                job.request_time
            )
            results[job_id] = worst_case
            
        return results
    
    def getWorstCaseForSingleJob(self, start_window, end_window, job_length_seconds):
        """
        Convenience function for analyzing a single job's worst-case carbon exposure.
        
        Args:
            start_window: earliest possible start time (seconds)
            end_window: latest possible end time (seconds)
            job_length_seconds: job duration in seconds
            
        Returns:
            Sum of worst N carbon intensities the job could experience
        """
        return self.getMaxCarbonIntensityInPeriod(start_window, end_window, job_length_seconds)

    def getCarbonAwareReward(self, scheduledJobs):
        """
        Calculate carbon-aware reward for post-scheduling evaluation.
        Dispatches to different reward calculation methods based on configuration.
        
        Args:
            scheduledJobs: list of jobs with scheduled_time, submit_time, request_time, power, carbon_consideration
        
        Returns:
            Carbon-aware reward (negative value, higher magnitude = worse)
        """
        if CARBON_REWARD_FUNCTION == 'emission_ratio':
            return self._getCarbonAwareReward_EmissionRatio(scheduledJobs)
        elif CARBON_REWARD_FUNCTION == 'co2_direct':
            return self._getCarbonAwareReward_CO2Direct(scheduledJobs)
        elif CARBON_REWARD_FUNCTION == 'fcfs_normalized':
            return self._getCarbonAwareReward_FCFSNormalized(scheduledJobs)
        else:
            # Default to emission_ratio if unknown function specified
            print(f"Warning: Unknown carbon reward function '{CARBON_REWARD_FUNCTION}', using 'emission_ratio'")
            return self._getCarbonAwareReward_EmissionRatio(scheduledJobs)

    def _getCarbonAwareReward_EmissionRatio(self, scheduledJobs):
        """
        Original emission ratio method: Compare actual vs worst-case emissions.
        Reward is based on the average of carbon-consideration-weighted emission ratios.
        Higher carbon consideration makes the job's performance impact the reward more.
        
        Args:
            scheduledJobs: list of jobs with scheduled_time, submit_time, request_time, power, carbon_consideration
        
        Returns:
            Carbon-aware reward (negative value, higher magnitude = worse)
        """
        if not scheduledJobs:
            return 0.0
        
        # Filter out unscheduled jobs
        valid_jobs = [job for job in scheduledJobs 
                     if hasattr(job, 'scheduled_time') and job.scheduled_time != -1]
        
        if not valid_jobs:
            return 0.0
        
        # Get worst-case carbon intensities for all jobs
        worst_case_intensities = self.getMaxCarbonIntensityFromJobs(valid_jobs)
        
        total_weighted_ratios = 0
        job_count = 0
        
        # Process each job individually
        for job in valid_jobs:
            if not hasattr(job, 'scheduled_time') or job.scheduled_time == -1:
                continue  # Skip unscheduled jobs
            
            # Job data
            job_id = getattr(job, 'job_id', 'unknown')
            start_time = job.scheduled_time
            end_time = start_time + job.request_time
            power = job.power  # watts
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            
            if carbon_consideration == 0:
                continue  # Skip jobs with no carbon consideration
            
            # Calculate actual carbon emissions for this job's scheduled time
            actual_emissions = self.getCarbonEmissions(power, start_time, end_time)
            
            # Get worst-case carbon emissions
            # getMaxCarbonIntensityInPeriod now returns weighted sum directly
            worst_case_intensity_weighted_sum = worst_case_intensities.get(job_id, 0)
            
            # Convert to emissions: weighted_sum represents total intensity exposure
            # for the job duration, so we multiply by power and convert to emissions
            duration_hours = (end_time - start_time) / 3600.0
            energy_kwh = (power / 1000.0) * duration_hours
            
            # The weighted sum already accounts for the job duration, so we need the average
            avg_worst_case_intensity = worst_case_intensity_weighted_sum / duration_hours
            worst_case_emissions = energy_kwh * avg_worst_case_intensity
            
            # Calculate ratio of actual to worst-case emissions
            if worst_case_emissions > 0:
                emission_ratio = actual_emissions / worst_case_emissions
            else:
                emission_ratio = 0.0
            
            # Weight the ratio by carbon consideration (higher consideration = more impact)
            weighted_ratio = emission_ratio * carbon_consideration
            total_weighted_ratios += weighted_ratio
            job_count += 1
        
        if job_count == 0:
            return 0.0
        
        # Calculate simple average of weighted ratios
        avg_weighted_ratio = total_weighted_ratios / job_count
        
        # Return negative reward (penalty) - higher carbon consideration jobs with worse ratios = worse reward
        return -avg_weighted_ratio

    def _getCarbonAwareReward_CO2Direct(self, scheduledJobs):
        """
        Direct CO2 method: Simple CO2 emissions multiplied by carbon consideration.
        Calculates actual CO2 emissions for each job and multiplies by its carbon consideration.
        
        Args:
            scheduledJobs: list of jobs with scheduled_time, submit_time, request_time, power, carbon_consideration
        
        Returns:
            Carbon-aware reward (negative value, higher magnitude = worse)
        """
        if not scheduledJobs:
            return 0.0
        
        # Filter out unscheduled jobs
        valid_jobs = [job for job in scheduledJobs 
                     if hasattr(job, 'scheduled_time') and job.scheduled_time != -1]
        
        if not valid_jobs:
            return 0.0
        
        total_weighted_co2 = 0.0
        
        for job in valid_jobs:
            start_time = job.scheduled_time
            end_time = start_time + job.request_time
            power = job.power
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            
            # Calculate actual CO2 emissions for this job
            co2_emissions = self.getCarbonEmissions(power, start_time, end_time)
            
            # Multiply by carbon consideration
            weighted_co2 = co2_emissions * carbon_consideration
            total_weighted_co2 += weighted_co2
        
        # Return negative (penalty) and scale to reasonable range (gCO2 to kgCO2)
        return -total_weighted_co2 / 100000.0

    def _getCarbonAwareReward_FCFSNormalized(self, scheduledJobs):
        """
        FCFS-normalized reward function that calculates percentage change in emissions 
        relative to FCFS baseline, weighted by carbon consideration.
        
        Equivalent to: -Σ(carbon_consideration * Δem_pct_per_job) / Σ(carbon_consideration)
        where Δem_pct_per_job = (em_cur_job - em_fcfs_job) / (em_fcfs_job + 1e-6)
        
        Args:
            scheduledJobs: list of jobs with scheduled_time, submit_time, request_time, power, carbon_consideration
        
        Returns:
            FCFS-normalized carbon reward weighted by carbon consideration (higher = better, positive = improvement over FCFS)
        """
        if not scheduledJobs:
            return 0.0
        
        # Filter out unscheduled jobs
        valid_jobs = [job for job in scheduledJobs 
                     if hasattr(job, 'scheduled_time') and job.scheduled_time != -1]
        
        if not valid_jobs:
            return 0.0
        
        # Calculate FCFS baseline schedule for comparison
        fcfs_schedule = self._simulate_fcfs_schedule(valid_jobs)
        
        total_weighted_delta = 0.0
        total_carbon_consideration = 0.0
        
        for job in valid_jobs:
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            
            # Skip jobs with no carbon consideration
            if carbon_consideration == 0:
                continue
            
            # Current schedule emissions for this job
            em_cur_job = self.getCarbonEmissions(job.power, job.scheduled_time, 
                                               job.scheduled_time + job.request_time)
            
            # FCFS schedule emissions for this job
            fcfs_start_time = fcfs_schedule.get(job.job_id, job.scheduled_time)
            em_fcfs_job = self.getCarbonEmissions(job.power, fcfs_start_time, 
                                                fcfs_start_time + job.request_time)
            
            # Calculate percentage change for this job
            Δem_pct_job = (em_cur_job - em_fcfs_job) / (em_fcfs_job + 1e-6)
            
            # Weight by carbon consideration
            weighted_delta = carbon_consideration * Δem_pct_job
            total_weighted_delta += weighted_delta
            total_carbon_consideration += carbon_consideration
        
        if total_carbon_consideration == 0:
            return 0.0
        
        # Calculate weighted average percentage change
        avg_weighted_delta = total_weighted_delta / total_carbon_consideration
        
        # Return negative of percentage change (higher = better, positive = improvement over FCFS)
        return -avg_weighted_delta
    
    def _simulate_fcfs_schedule(self, jobs):
        """
        Simulate FCFS scheduling to get start times for each job.
        
        Args:
            jobs: list of jobs to schedule
            
        Returns:
            Dictionary mapping job_id to FCFS start_time
        """
        # Sort jobs by submit time (FCFS order)
        fcfs_jobs = sorted(jobs, key=lambda j: j.submit_time)
        
        current_time = min(job.submit_time for job in fcfs_jobs) if fcfs_jobs else 0
        fcfs_schedule = {}
        
        for job in fcfs_jobs:
            # Job starts when it arrives or when current time allows
            fcfs_start_time = max(current_time, job.submit_time)
            fcfs_schedule[job.job_id] = fcfs_start_time
            
            # Update current time (assuming infinite resources for simplicity)
            current_time = fcfs_start_time + job.request_time
        
        return fcfs_schedule

    def getCarbonMetrics(self, scheduledJobs):
        """
        Calculate comprehensive carbon metrics for post-scheduling analysis.
        
        Args:
            scheduledJobs: list of jobs with timing, power, and carbon consideration data
            
        Returns:
            Dictionary with carbon metrics:
            - total_emissions: total CO2 emissions (gCO2eq)
            - total_energy: total energy consumption (kWh) 
            - avg_carbon_intensity: average carbon intensity (gCO2eq/kWh)
            - weighted_emissions: carbon-consideration-weighted emissions
            - per_job_metrics: list of per-job carbon data
        """
        if not scheduledJobs:
            return {
                'total_emissions': 0,
                'total_energy': 0, 
                'avg_carbon_intensity': 0,
                'weighted_emissions': 0,
                'per_job_metrics': []
            }
        
        total_emissions = 0
        total_energy = 0
        total_weighted_emissions = 0
        per_job_metrics = []
        
        for job in scheduledJobs:
            if not hasattr(job, 'scheduled_time') or job.scheduled_time == -1:
                continue
                
            # Job data
            start_time = job.scheduled_time
            end_time = start_time + job.request_time
            power = job.power
            carbon_consideration = getattr(job, 'carbon_consideration', 0)
            
            # Calculate metrics for this job
            duration_hours = (end_time - start_time) / 3600.0
            energy_kwh = (power / 1000.0) * duration_hours
            emissions = self.getCarbonEmissions(power, start_time, end_time)
            
            # Carbon consideration directly scales the penalty
            weighted_emissions = emissions * carbon_consideration
            
            # Accumulate totals
            total_energy += energy_kwh
            total_emissions += emissions
            total_weighted_emissions += weighted_emissions
            
            # Store per-job metrics
            per_job_metrics.append({
                'job_id': getattr(job, 'job_id', 'unknown'),
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': duration_hours,
                'power_watts': power,
                'energy_kwh': energy_kwh,
                'emissions_gco2eq': emissions,
                'carbon_consideration': carbon_consideration,
                'weighted_emissions': weighted_emissions
            })
        
        # Calculate overall metrics
        avg_carbon_intensity = total_emissions / total_energy if total_energy > 0 else 0
        
        return {
            'total_emissions': total_emissions,
            'total_energy': total_energy,
            'avg_carbon_intensity': avg_carbon_intensity, 
            'weighted_emissions': total_weighted_emissions,
            'per_job_metrics': per_job_metrics
        }
