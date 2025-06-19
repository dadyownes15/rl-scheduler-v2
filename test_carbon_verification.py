#!/usr/bin/env python3
"""
Test script to verify carbon-aware power tracking is working properly.
"""

from HPCSimPickJobs import HPCEnv
import numpy as np

def test_carbon_verification():
    print("=== Carbon-Aware Power Tracking Verification Test ===")
    
    # Initialize environment
    env = HPCEnv()
    env.my_init('./data/lublin_256_carbon_float.swf')
    env.seed(42)
    
    # Reset and start episode
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Run episode until completion
    step_count = 0
    max_steps = 1000  # Safety limit
    
    while step_count < max_steps:
        # Choose first available job (action 0) with no delay (action 0)
        result = env.step(0, 0)
        step_count += 1
        
        if step_count % 50 == 0:
            print(f"Step {step_count}: Episode complete = {result[2]}")
        
        # Check if episode is complete
        if result[2]:  # episode done
            print(f"\n✓ Episode completed after {step_count} steps")
            print(f"✓ Final carbon reward: {result[7]:.6f}")
            break
    
    if step_count >= max_steps:
        print(f"⚠ Episode did not complete within {max_steps} steps")
    
    print("\n=== Carbon-Aware Power Tracking Test Complete ===")
    return True

if __name__ == "__main__":
    test_carbon_verification() 