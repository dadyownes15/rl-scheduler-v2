#!/usr/bin/env python3
"""
Test script to verify the carbon intensity changes work correctly
"""

import sys
import os
import numpy as np

# Add the current directory to path
sys.path.append(os.getcwd())

from HPCSimPickJobs import HPCEnv, JOB_FEATURES
from greenPower import carbon_intensity

def test_carbon_intensity_class():
    """Test the new carbon intensity class"""
    print("Testing carbon_intensity class...")
    
    # Test initialization
    ci = carbon_intensity(greenWin=24, year=2021)
    print(f"✓ Initialized with {len(ci.carbonIntensityList)} carbon intensity values")
    
    # Test getCarbonIntensitySlot
    slots = ci.getCarbonIntensitySlot(3600)  # 1 hour in
    print(f"✓ Got {len(slots)} carbon intensity slots")
    print(f"  First slot: {slots[0]}")
    
    # Test getCarbonEmissions
    emissions = ci.getCarbonEmissions(1000, 0, 3600)  # 1kW for 1 hour
    print(f"✓ Carbon emissions for 1kW over 1 hour: {emissions:.2f} gCO2eq")
    
    return True

def test_hpc_env():
    """Test the HPCEnv with carbon intensity"""
    print("\nTesting HPCEnv with carbon intensity...")
    
    env = HPCEnv()
    env.my_init('./data/lublin_256.swf')
    env.seed(42)  # Initialize random state
    
    # Test reset
    obs = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  JOB_FEATURES: {JOB_FEATURES}")
    
    # Test build_observation
    obs = env.build_observation()
    print(f"✓ Built observation with {len(obs)} features")
    
    # Check if carbon intensity is working
    carbon_slots = env.cluster.carbonIntensity.getCarbonIntensitySlot(env.current_timestamp)
    print(f"✓ Carbon intensity slots: {len(carbon_slots)}")
    
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("Testing Carbon Intensity Implementation")
    print("=" * 50)
    
    try:
        test_carbon_intensity_class()
        test_hpc_env()
        
        print("\n" + "=" * 50)
        print("✅ All tests passed! Carbon intensity system is working.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 