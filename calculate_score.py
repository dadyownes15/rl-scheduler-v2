#!/usr/bin/env python3
"""
Calculate Score Based on OLS Regression Coefficients

This script calculates a score based on the ratio of carbon_consideration coefficients
from two OLS regressions:
1. Carbon emissions regression
2. Wait time regression

The score represents the trade-off between carbon awareness and wait time impact.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import argparse
import os
from datetime import datetime

def load_and_prepare_data(job_details_file, carbon_window_file=None):
    """
    Load job details and optional carbon window data
    
    Args:
        job_details_file: Path to job_details_enhanced.csv
        carbon_window_file: Optional path to carbon window data
    
    Returns:
        DataFrame with prepared data
    """
    print(f"Loading job details from: {job_details_file}")
    df = pd.read_csv(job_details_file)
    print(f"Loaded {len(df)} job records")
    
    # If carbon window file is provided, merge it
    if carbon_window_file and os.path.exists(carbon_window_file):
        print(f"Loading carbon window data from: {carbon_window_file}")
        carbon_df = pd.read_csv(carbon_window_file)
        # Merge on job_id or timestamp - adjust based on actual data structure
        if 'job_id' in carbon_df.columns:
            df = df.merge(carbon_df, on='job_id', how='left')
        print(f"Merged carbon window data")
    
    return df

def calculate_carbon_intensity_baseline(df):
    """
    Calculate baseline carbon intensity as average over the viewable window
    
    Args:
        df: DataFrame with job data
    
    Returns:
        Series with carbon intensity baseline for each job
    """
    # If we have carbon window data, use it
    if 'carbon_intensity_avg' in df.columns:
        return df['carbon_intensity_avg']
    
    # Otherwise, calculate a simple baseline
    # This would need to be updated when carbon window data is available
    print("Warning: No carbon window data found, using simple baseline")
    return pd.Series(350.0, index=df.index)  # Default carbon intensity

def run_carbon_emissions_regression(df):
    """
    Run OLS regression for carbon emissions
    
    Regressors:
    - carbon_consideration
    - request_time * request_processors (interaction)
    - carbon_consideration * request_time * request_processors (triple interaction)
    - carbon_intensity_baseline
    
    Args:
        df: DataFrame with job data
    
    Returns:
        OLS regression results
    """
    print("\n" + "="*60)
    print("CARBON EMISSIONS REGRESSION")
    print("="*60)
    
    # Create features
    df['runtime_x_processors'] = df['request_time'] * df['request_processors']
    df['carbon_x_runtime_x_processors'] = (df['carbon_consideration'] * 
                                          df['request_time'] * 
                                          df['request_processors'])
    df['carbon_intensity_baseline'] = calculate_carbon_intensity_baseline(df)
    
    # Define feature matrix
    features = [
        'carbon_consideration',
        'runtime_x_processors', 
        'carbon_x_runtime_x_processors',
        'carbon_intensity_baseline'
    ]
    
    X = df[features].copy()
    y = df['carbon_emissions']
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f} (p-value: {model.f_pvalue:.4e})")
    
    # Print coefficients
    print("\nCoefficients:")
    for i, feature in enumerate(['const'] + features):
        coef = model.params[i]
        pval = model.pvalues[i]
        print(f"  {feature:30s}: {coef:10.6f} (p={pval:.4f})")
    
    return model, X, y

def run_wait_time_regression(df):
    """
    Run OLS regression for wait time
    
    Regressors:
    - carbon_consideration
    - request_time * request_processors (interaction)
    - carbon_consideration * request_time * request_processors (triple interaction)
    - carbon_intensity_baseline
    - queue_length_at_submission (additional for wait time)
    
    Args:
        df: DataFrame with job data
    
    Returns:
        OLS regression results
    """
    print("\n" + "="*60)
    print("WAIT TIME REGRESSION")
    print("="*60)
    
    # Features already created in carbon regression
    features = [
        'carbon_consideration',
        'runtime_x_processors',
        'carbon_x_runtime_x_processors', 
        'carbon_intensity_baseline',
        'queue_length_at_submission'  # Additional feature for wait time
    ]
    
    X = df[features].copy()
    y = df['wait_time']
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"R-squared: {model.rsquared:.4f}")
    print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f} (p-value: {model.f_pvalue:.4e})")
    
    # Print coefficients
    print("\nCoefficients:")
    for i, feature in enumerate(['const'] + features):
        coef = model.params[i]
        pval = model.pvalues[i]
        print(f"  {feature:30s}: {coef:10.6f} (p={pval:.4f})")
    
    return model, X, y

def calculate_score(carbon_model, wait_model):
    """
    Calculate score as negative ratio of carbon_consideration coefficients
    
    Score = -(carbon_consideration_coef_carbon / carbon_consideration_coef_wait)
    
    Args:
        carbon_model: OLS results for carbon emissions
        wait_model: OLS results for wait time
    
    Returns:
        float: Calculated score
    """
    print("\n" + "="*60)
    print("SCORE CALCULATION")
    print("="*60)
    
    # Get carbon_consideration coefficients
    carbon_coef = carbon_model.params['carbon_consideration']
    wait_coef = wait_model.params['carbon_consideration']
    
    print(f"Carbon consideration coefficient (carbon emissions): {carbon_coef:.6f}")
    print(f"Carbon consideration coefficient (wait time): {wait_coef:.6f}")
    
    # Calculate score
    if abs(wait_coef) < 1e-10:
        print("Warning: Wait time coefficient near zero, score may be unstable")
        score = float('inf') if carbon_coef != 0 else 0
    else:
        score = -(carbon_coef / wait_coef)
    
    print(f"Score = -(carbon_coef / wait_coef) = -({carbon_coef:.6f} / {wait_coef:.6f}) = {score:.6f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if score > 0:
        print(f"  Positive score ({score:.3f}): Carbon consideration reduces emissions more than it increases wait time")
    elif score < 0:
        print(f"  Negative score ({score:.3f}): Carbon consideration increases wait time more than it reduces emissions")
    else:
        print(f"  Zero score: No net trade-off effect")
    
    return score

def save_results(carbon_model, wait_model, score, output_file):
    """
    Save detailed results to file
    
    Args:
        carbon_model: OLS results for carbon emissions
        wait_model: OLS results for wait time
        score: Calculated score
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SCORE CALCULATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Score: {score:.6f}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CARBON EMISSIONS REGRESSION RESULTS:\n")
        f.write("="*50 + "\n")
        f.write(str(carbon_model.summary()))
        f.write("\n\n")
        
        f.write("WAIT TIME REGRESSION RESULTS:\n")
        f.write("="*50 + "\n")
        f.write(str(wait_model.summary()))
        f.write("\n\n")
        
        # Key coefficients comparison
        f.write("KEY COEFFICIENTS COMPARISON:\n")
        f.write("="*30 + "\n")
        f.write(f"Carbon consideration (carbon emissions): {carbon_model.params['carbon_consideration']:.6f}\n")
        f.write(f"Carbon consideration (wait time): {wait_model.params['carbon_consideration']:.6f}\n")
        f.write(f"Score: {score:.6f}\n")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate Score from OLS Regression Coefficients')
    parser.add_argument('--job_details', required=True, help='Path to job_details_enhanced.csv')
    parser.add_argument('--carbon_window', help='Path to carbon window data (optional)')
    parser.add_argument('--output', help='Output file for detailed results')
    
    args = parser.parse_args()
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(args.job_details, args.carbon_window)
        
        # Check required columns
        required_cols = ['carbon_consideration', 'request_time', 'request_processors', 
                        'carbon_emissions', 'wait_time', 'queue_length_at_submission']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Run regressions
        carbon_model, carbon_X, carbon_y = run_carbon_emissions_regression(df)
        wait_model, wait_X, wait_y = run_wait_time_regression(df)
        
        # Calculate score
        score = calculate_score(carbon_model, wait_model)
        
        # Save results if output file specified
        if args.output:
            save_results(carbon_model, wait_model, score, args.output)
            print(f"\nDetailed results saved to: {args.output}")
        
        print(f"\n" + "="*60)
        print(f"FINAL SCORE: {score:.6f}")
        print("="*60)
        
        return score
        
    except Exception as e:
        print(f"Error during score calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 