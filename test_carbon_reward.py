from greenPower import carbon_intensity

# Test the updated carbon reward function
carbon_int = carbon_intensity(greenWin=48, year=2021)

# Create mock scheduled jobs for testing
class MockJob:
    def __init__(self, job_id, scheduled_time, request_time, power, carbon_consideration):
        self.job_id = job_id
        self.scheduled_time = scheduled_time
        self.request_time = request_time
        self.power = power
        self.carbon_consideration = carbon_consideration

jobs = [
    MockJob(1, 0, 3600, 1000, 0.5),      # 1 hour, 1kW, medium carbon concern (0.5)
    MockJob(2, 3600, 1800, 500, 1.0),   # 30min, 0.5kW, high carbon concern (1.0)
    MockJob(3, 5400, 3600, 2000, 0.0),  # 1 hour, 2kW, no carbon concern (0.0)
]

# Test the carbon reward calculation
reward = carbon_int.getCarbonAwareReward(jobs)
print(f'Carbon reward: {reward:.6f}')

# Test the comprehensive metrics
metrics = carbon_int.getCarbonMetrics(jobs)
print(f'Total emissions: {metrics["total_emissions"]:.2f} gCO2eq')
print(f'Total energy: {metrics["total_energy"]:.2f} kWh')
print(f'Average carbon intensity: {metrics["avg_carbon_intensity"]:.2f} gCO2eq/kWh')
print(f'Weighted emissions: {metrics["weighted_emissions"]:.2f} gCO2eq')
print(f'Number of jobs processed: {len(metrics["per_job_metrics"])}')

# Show per-job breakdown
print('\nPer-job breakdown:')
for job_metrics in metrics["per_job_metrics"]:
    print(f'Job {job_metrics["job_id"]}: carbon_consideration={job_metrics["carbon_consideration"]}, '
          f'emissions={job_metrics["emissions_gco2eq"]:.1f}gCO2eq, '
          f'weighted_emissions={job_metrics["weighted_emissions"]:.1f}gCO2eq') 