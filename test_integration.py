from HPCSimPickJobs import *
from greenPower import carbon_intensity

# Mock job class for testing
class MockJob:
    def __init__(self, job_id, submit_time, scheduled_time, request_time, power=1000, carbon_consideration=1.0):
        self.job_id = job_id
        self.submit_time = submit_time
        self.scheduled_time = scheduled_time
        self.request_time = request_time
        self.power = power
        self.carbon_consideration = carbon_consideration

# Test integration with HPCEnv
env = HPCEnv()
env.my_init('./data/lublin_256_carbon_float.swf')
env.seed(42)

print("Testing carbon reward integration with HPCEnv...")

# Reset environment
obs = env.reset()
print(f"Episode start offset: {env.episode_start_hour_offset}")

# Take one step
action1, action2 = 0, 0
result = env.step(action1, action2)

print(f"Step result length: {len(result)}")
if len(result) >= 8:
    carbon_reward = result[7]
    print(f"Carbon reward from step: {carbon_reward:.4f}")
else:
    print("Step completed but no carbon reward returned")

print("Integration test completed!")

print("\n--- Testing Worst-Case Window Update ---")
print("Worst-case window changed to 24 hours from scheduled time")

# Create a job with specific timing
job = MockJob(
    job_id="window_test",
    submit_time=1*3600,      # 1 AM
    scheduled_time=5*3600,   # 5 AM (4h wait)
    request_time=2*3600,     # 2h duration
    power=1500,
    carbon_consideration=0.8
)

print(f"Job: submit={job.submit_time/3600:.1f}h, schedule={job.scheduled_time/3600:.1f}h, duration={job.request_time/3600:.1f}h")

# New worst-case window: 24 hours starting from scheduled_time (5h to 29h)
worst_case_window_start = job.scheduled_time
worst_case_window_end = job.scheduled_time + (24 * 3600)
print(f"Worst-case window: {worst_case_window_start/3600:.1f}h to {worst_case_window_end/3600:.1f}h (24h from start)")

carbon = carbon_intensity(greenWin=5, year=2021)
carbon.setStartOffset(0)
worst_case = carbon.getMaxCarbonIntensityFromJobs([job])
print(f"Worst-case carbon intensity: {worst_case['window_test']:.2f} gCO2eq/kWh")

# Test with different scheduled time
late_job = MockJob(
    job_id="late_test",
    submit_time=2*3600,      # 2 AM
    scheduled_time=15*3600,  # 3 PM (13h wait)
    request_time=3*3600,     # 3h duration
    power=1500,
    carbon_consideration=0.8
)

# Worst-case window: 24 hours starting from 15h (15h to 39h, wrapping to next day)
worst_case_late = carbon.getMaxCarbonIntensityFromJobs([late_job])
print(f"Late job worst-case (15h to 39h): {worst_case_late['late_test']:.2f} gCO2eq/kWh") 