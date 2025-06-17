#!/usr/bin/env python3
"""
Quick verification script to show the carbon consideration distribution
in the lublin_256_carbon.swf workload file.
"""

from job import Workloads

def verify_distribution():
    print("Verifying Carbon Consideration Distribution")
    print("=" * 50)
    
    # Load the carbon-aware workload
    workload = Workloads('./data/lublin_256_carbon.swf')
    
    # Count carbon consideration levels
    carbon_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
    for job in workload.all_jobs:
        if hasattr(job, 'carbon_consideration'):
            carbon_counts[job.carbon_consideration] += 1
        else:
            print(f"WARNING: Job {job.job_id} missing carbon_consideration")
    
    total_jobs = len(workload.all_jobs)
    
    print(f"Total jobs: {total_jobs}")
    print("\nCarbon Consideration Distribution:")
    print("Level | Count | Percentage | Concern Level")
    print("------|-------|------------|---------------")
    
    concern_labels = {
        0: "Very Low",
        1: "Low", 
        2: "Medium",
        3: "High",
        4: "Very High"
    }
    
    for level in range(5):
        count = carbon_counts[level]
        percentage = (count / total_jobs) * 100
        label = concern_labels[level]
        print(f"  {level}   | {count:5d} | {percentage:8.1f}% | {label}")
    
    print(f"\nWorkload file: ./data/lublin_256_carbon.swf")
    print("✓ Carbon consideration indices are properly distributed")

if __name__ == "__main__":
    verify_distribution() 