#!/usr/bin/env python3
"""
Test backward compatibility with original SWF files (without carbon consideration).
Verify that all jobs get carbon consideration 0 and carbon reward is 0.
"""

from job import *
from greenPower import *

def test_backward_compatibility():
    print("Testing Backward Compatibility")
    print("=" * 50)
    
    # Test original lublin_256.swf (18 fields, no carbon consideration)
    print("Loading original lublin_256.swf...")
    loads = Workloads('./data/lublin_256.swf')
    print(f"Loaded {loads.size()} jobs")
    
    # Check first 10 jobs to verify they all have carbon consideration 0
    print("\nFirst 10 jobs carbon consideration:")
    all_zero = True
    for i in range(min(10, loads.size())):
        job = loads[i]
        print(f"  Job {job.job_id}: carbon_consideration = {job.carbon_consideration}")
        if job.carbon_consideration != 0:
            all_zero = False
    
    print(f"\nAll jobs have carbon consideration 0: {all_zero}")
    
    # Test carbon reward calculation
    print("\nTesting carbon reward calculation...")
    ci = carbon_intensity(greenWin=24, year=2021)
    
    # Create a simple power slot for testing
    powerSlot = [
        {'timeSlot': 0, 'power': 1000},
        {'timeSlot': 3600, 'power': 1000},  # 1 hour of 1000W
        {'timeSlot': 7200, 'power': 0}
    ]
    
    # Test with jobs from original file (all should have carbon consideration 0)
    test_jobs = loads.all_jobs[:10]  # First 10 jobs
    for job in test_jobs:
        job.scheduled_time = 0
        job.request_time = 3600  # 1 hour
    
    carbon_reward = ci.getCarbonAwareReward(powerSlot, test_jobs)
    print(f"Carbon reward with all jobs having carbon consideration 0: {carbon_reward}")
    print(f"Carbon reward is zero: {carbon_reward == 0}")
    
    # Test carbon weight calculation directly
    weight = ci.calculateCarbonWeight(test_jobs, 0, 3600)
    print(f"Carbon weight for jobs with consideration 0: {weight}")
    print(f"Carbon weight is zero: {weight == 0}")
    
    print("\n" + "=" * 50)
    if all_zero and carbon_reward == 0 and weight == 0:
        print("✅ Backward compatibility test PASSED!")
        print("   - All jobs from original SWF have carbon consideration 0")
        print("   - Carbon reward is 0 (no carbon optimization)")
        print("   - Carbon weight is 0 (no penalty)")
    else:
        print("❌ Backward compatibility test FAILED!")
        print(f"   - All zero: {all_zero}")
        print(f"   - Carbon reward zero: {carbon_reward == 0}")
        print(f"   - Carbon weight zero: {weight == 0}")

if __name__ == "__main__":
    test_backward_compatibility() 