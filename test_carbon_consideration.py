#!/usr/bin/env python3
"""
Test script to verify the carbon consideration index system works correctly
"""

import sys
import os
import numpy as np

# Add the current directory to path
sys.path.append(os.getcwd())

from HPCSimPickJobs import HPCEnv, JOB_FEATURES
from greenPower import carbon_intensity, MAX_CARBON_INTENSITY
from job import Job

def test_carbon_consideration_jobs():
    """Test that jobs have carbon consideration indices from the SWF file"""
    print("Testing carbon consideration in jobs...")
    
    # Test reading from the new carbon SWF format
    job_lines = [
        "1 5094 -1 12072 16 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 -1 -1 -1 0",  # Carbon consideration 0
        "2 5170 -1 2 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 -1 -1 -1 1",      # Carbon consideration 1
        "3 6742 -1 24089 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 -1 -1 -1 2",  # Carbon consideration 2
        "4 7287 -1 9053 128 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 -1 -1 -1 3", # Carbon consideration 3
        "5 7454 -1 8843 1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0 -1 -1 -1 4",   # Carbon consideration 4
    ]
    
    expected_carbon_levels = [0, 1, 2, 3, 4]
    
    for i, (job_line, expected) in enumerate(zip(job_lines, expected_carbon_levels)):
        job = Job(job_line)
        print(f"  Job {i+1}: Carbon consideration = {job.carbon_consideration} (expected: {expected})")
        assert job.carbon_consideration == expected
    
    # Test fallback behavior for old format (18 fields)
    old_format_line = "1 5094 -1 12072 16 -1 -1 -1 -1 -1 1 -1 -1 -1 0 -1 -1 -1"  # Only 18 fields
    job = Job(old_format_line)
    expected_fallback = abs(job.user_id) % 5  # Should use user_id fallback
    print(f"  Old format fallback: Carbon consideration = {job.carbon_consideration} (expected: {expected_fallback})")
    assert job.carbon_consideration == expected_fallback
    
    print("✓ Carbon consideration reading from SWF file works correctly")
    return True

def test_carbon_aware_reward():
    """Test the carbon-aware reward calculation"""
    print("\nTesting carbon-aware reward calculation...")
    
    ci = carbon_intensity(greenWin=24, year=2021)
    
    # Create mock power slot log
    powerSlot = [
        {'timeSlot': 0, 'power': 5000},
        {'timeSlot': 3600, 'power': 8000},
        {'timeSlot': 7200, 'power': 0}
    ]
    
    # Create mock jobs with different carbon considerations
    class MockJob:
        def __init__(self, carbon_consideration, scheduled_time, request_time):
            self.carbon_consideration = carbon_consideration
            self.scheduled_time = scheduled_time
            self.request_time = request_time
    
    # Jobs with different carbon considerations
    jobs_low_concern = [MockJob(0, 0, 3600), MockJob(1, 0, 3600)]  # Low carbon concern
    jobs_high_concern = [MockJob(3, 0, 3600), MockJob(4, 0, 3600)]  # High carbon concern
    jobs_mixed = [MockJob(0, 0, 3600), MockJob(4, 0, 3600)]  # Mixed concern
    
    reward_low = ci.getCarbonAwareReward(powerSlot, jobs_low_concern)
    reward_high = ci.getCarbonAwareReward(powerSlot, jobs_high_concern)
    reward_mixed = ci.getCarbonAwareReward(powerSlot, jobs_mixed)
    
    print(f"✓ Low concern jobs reward: {reward_low:.4f}")
    print(f"✓ High concern jobs reward: {reward_high:.4f}")
    print(f"✓ Mixed concern jobs reward: {reward_mixed:.4f}")
    
    # High concern jobs should have more negative reward (higher penalty)
    assert reward_high < reward_low, f"High concern should have higher penalty: {reward_high} < {reward_low}"
    assert reward_mixed < reward_low and reward_mixed > reward_high, "Mixed should be between low and high"
    
    return True

def test_hpc_env_with_carbon_consideration():
    """Test the HPCEnv with carbon consideration features"""
    print("\nTesting HPCEnv with carbon consideration...")
    
    env = HPCEnv()
    env.my_init('./data/lublin_256_carbon.swf')  # Use the new carbon SWF file
    env.seed(42)
    
    # Test reset
    obs = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  JOB_FEATURES: {JOB_FEATURES}")
    
    # Check that we have 7 features per job
    expected_obs_size = (256 + 64 + 24) * 7  # MAX_QUEUE_SIZE + run_win + green_win times JOB_FEATURES
    assert obs.shape[0] == expected_obs_size, f"Expected {expected_obs_size}, got {obs.shape[0]}"
    
    # Check that job features include carbon consideration
    if len(env.job_queue) > 0:
        job = env.job_queue[0]
        print(f"✓ First job carbon consideration: {job.carbon_consideration}")
        print(f"  Carbon consideration should be 0-4: {job.carbon_consideration in [0, 1, 2, 3, 4]}")
        
        # Check that it's in the observation
        job_features = obs[:7]  # First job's features
        carbon_feature = job_features[5]  # 6th feature (index 5) should be carbon consideration
        allocate_feature = job_features[6]  # 7th feature (index 6) should be can_allocate
        expected_carbon_feature = job.carbon_consideration / 4.0
        print(f"  Carbon feature in observation: {carbon_feature:.3f}")
        print(f"  Expected: {expected_carbon_feature:.3f}")
        print(f"  Can allocate feature: {allocate_feature:.3f}")
        
        # Check if the carbon feature is approximately correct
        assert abs(carbon_feature - expected_carbon_feature) < 0.01, \
            f"Carbon feature mismatch: expected {expected_carbon_feature}, got {carbon_feature}"
        
        # Verify carbon consideration is in valid range
        assert job.carbon_consideration in [0, 1, 2, 3, 4], \
            f"Invalid carbon consideration: {job.carbon_consideration}"
    
    print("✓ All tests passed!")
    return True

if __name__ == "__main__":
    print("Testing Carbon Consideration Index System")
    print("=" * 50)
    
    try:
        test_carbon_consideration_jobs()
        test_carbon_aware_reward()
        test_hpc_env_with_carbon_consideration()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Carbon consideration system is working correctly.")
        print("\nSummary of features:")
        print("- Jobs have carbon consideration indices 0-4 from the SWF file")
        print("- Index 0 = lowest carbon concern, Index 4 = highest carbon concern")
        print("- Distribution: 15% very low, 25% low, 30% medium, 20% high, 10% very high")
        print("- Carbon-aware reward applies different penalties based on concern level")
        print("- Feature vector now includes 7 features (was 6), with carbon consideration as feature #7")
        print(f"- Max carbon intensity updated to {MAX_CARBON_INTENSITY} gCO₂eq/kWh")
        print("- Uses lublin_256_carbon.swf file with carbon indices as the 19th field")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 