#!/usr/bin/env python3
"""
OLS Regression Analysis for Job Details Enhanced CSV

This script performs Ordinary Least Squares regression analysis on job scheduling data
to understand the relationships between job characteristics and both carbon emissions 
and wait times. Results are saved to a TXT file in the experiment folder.

Features analyzed:
- request_time: Job runtime request in seconds
- request_processors: Number of processors requested
- carbon_consideration: Carbon awareness factor (0-1)
- queue_length_at_submission: Number of jobs in queue when submitted
- power: Power consumption in watts
- wait_time: Time waited in queue (seconds)

Target variables:
- carbon_emissions: Actual carbon emissions
- wait_time: Queue waiting time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_file, output_file):
    """Load CSV data and prepare for analysis"""
    output_file.write(f"Loading data from {csv_file}...\n")
    df = pd.read_csv(csv_file)
    output_file.write(f"Loaded {len(df)} records\n")
    
    # Display basic info about the dataset
    output_file.write("\nDataset Info:\n")
    output_file.write(f"Shape: {df.shape}\n")
    output_file.write(f"Columns: {list(df.columns)}\n")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        output_file.write("\nMissing values:\n")
        for col, count in missing_values[missing_values > 0].items():
            output_file.write(f"{col}: {count}\n")
    else:
        output_file.write("\nNo missing values found\n")
    
    # Basic statistics
    output_file.write("\nBasic statistics for key columns:\n")
    key_cols = ['request_time', 'request_processors', 'carbon_consideration', 
                'queue_length_at_submission', 'power', 'wait_time', 'carbon_emissions']
    output_file.write(str(df[key_cols].describe()))
    output_file.write("\n")
    
    return df

def create_feature_matrix(df, output_file):
    """Create feature matrix with base features and interactions"""
    output_file.write("\nCreating feature matrix...\n")
    
    # Base features
    features = [
        'request_time',
        'request_processors', 
        'carbon_consideration',
        'queue_length_at_submission',
        'power',
        'wait_time'
    ]
    
    # Create base feature matrix
    X_base = df[features].copy()
    
    # Create meaningful interactions based on domain knowledge
    interactions = {}
    
    # 1. Power-related interactions
    interactions['power_x_runtime'] = df['power'] * df['request_time']
    interactions['power_x_processors'] = df['power'] * df['request_processors']
    
    # 2. Queue dynamics interactions
    interactions['queue_x_processors'] = df['queue_length_at_submission'] * df['request_processors']
    interactions['queue_x_runtime'] = df['queue_length_at_submission'] * df['request_time']
    
    # 3. Carbon consideration interactions
    interactions['carbon_x_power'] = df['carbon_consideration'] * df['power']
    interactions['carbon_x_runtime'] = df['carbon_consideration'] * df['request_time']
    interactions['carbon_x_queue'] = df['carbon_consideration'] * df['queue_length_at_submission']
    
    # 4. Resource utilization interactions
    interactions['processors_x_runtime'] = df['request_processors'] * df['request_time']
    interactions['wait_x_runtime'] = df['wait_time'] * df['request_time']
    interactions['wait_x_processors'] = df['wait_time'] * df['request_processors']
    
    # 5. Efficiency ratios (log transforms to handle potential non-linearity)
    interactions['log_power_per_processor'] = np.log1p(df['power'] / (df['request_processors'] + 1))
    interactions['log_wait_per_queue'] = np.log1p(df['wait_time'] / (df['queue_length_at_submission'] + 1))
    
    # Combine base features with interactions
    X_interactions = pd.DataFrame(interactions)
    X_full = pd.concat([X_base, X_interactions], axis=1)
    
    output_file.write(f"Created {len(X_full.columns)} features:\n")
    output_file.write("Base features: " + str(features) + "\n")
    output_file.write("Interaction features: " + str(list(interactions.keys())) + "\n")
    
    return X_full, features, list(interactions.keys())

def perform_regression_analysis(X, y, target_name, feature_names, output_file):
    """Perform comprehensive OLS regression analysis"""
    output_file.write(f"\n{'='*60}\n")
    output_file.write(f"OLS REGRESSION ANALYSIS FOR {target_name.upper()}\n")
    output_file.write(f"{'='*60}\n")
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X_with_const).fit()
    
    # Write detailed results
    output_file.write("\nREGRESSION RESULTS:\n")
    output_file.write(str(model.summary()))
    output_file.write("\n")
    
    # Extract key metrics
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_statistic = model.fvalue
    f_pvalue = model.f_pvalue
    
    output_file.write(f"\nKEY METRICS:\n")
    output_file.write(f"R-squared: {r_squared:.4f}\n")
    output_file.write(f"Adjusted R-squared: {adj_r_squared:.4f}\n")
    output_file.write(f"F-statistic: {f_statistic:.4f}\n")
    output_file.write(f"F-statistic p-value: {f_pvalue:.4e}\n")
    
    # Feature importance (absolute coefficients)
    coefficients = model.params[1:]  # Exclude constant
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients.values,
        'abs_coefficient': np.abs(coefficients.values),
        'p_value': model.pvalues[1:].values
    }).sort_values('abs_coefficient', ascending=False)
    
    output_file.write(f"\nFEATURE IMPORTANCE (Top 10):\n")
    output_file.write(feature_importance.head(10).to_string(index=False))
    output_file.write("\n")
    
    # Significant features (p < 0.05)
    significant_features = feature_importance[feature_importance['p_value'] < 0.05]
    output_file.write(f"\nSIGNIFICANT FEATURES (p < 0.05): {len(significant_features)}\n")
    output_file.write(significant_features.to_string(index=False))
    output_file.write("\n")
    
    return model, feature_importance

def generate_summary_report(df, emissions_model, wait_time_model, feature_importance_emissions, feature_importance_wait, output_file):
    """Generate a comprehensive summary report"""
    output_file.write("\n" + "="*80 + "\n")
    output_file.write("COMPREHENSIVE REGRESSION ANALYSIS SUMMARY REPORT\n")
    output_file.write("="*80 + "\n")
    
    output_file.write(f"\nDATASET OVERVIEW:\n")
    output_file.write(f"- Total jobs analyzed: {len(df):,}\n")
    output_file.write(f"- Features used: {len(feature_importance_emissions)}\n")
    output_file.write(f"- Date range: {df['submit_time'].min()} to {df['submit_time'].max()}\n")
    
    output_file.write(f"\nTARGET VARIABLE STATISTICS:\n")
    output_file.write(f"Carbon Emissions:\n")
    output_file.write(f"  - Mean: {df['carbon_emissions'].mean():.4f}\n")
    output_file.write(f"  - Median: {df['carbon_emissions'].median():.4f}\n")
    output_file.write(f"  - Std Dev: {df['carbon_emissions'].std():.4f}\n")
    output_file.write(f"  - Range: {df['carbon_emissions'].min():.4f} to {df['carbon_emissions'].max():.4f}\n")
    
    output_file.write(f"\nWait Time:\n")
    output_file.write(f"  - Mean: {df['wait_time'].mean():.0f} seconds ({df['wait_time'].mean()/3600:.1f} hours)\n")
    output_file.write(f"  - Median: {df['wait_time'].median():.0f} seconds ({df['wait_time'].median()/3600:.1f} hours)\n")
    output_file.write(f"  - Std Dev: {df['wait_time'].std():.0f} seconds\n")
    output_file.write(f"  - Range: {df['wait_time'].min():.0f} to {df['wait_time'].max():.0f} seconds\n")
    
    output_file.write(f"\nMODEL PERFORMANCE:\n")
    output_file.write(f"Carbon Emissions Model:\n")
    output_file.write(f"  - R²: {emissions_model.rsquared:.4f}\n")
    output_file.write(f"  - Adjusted R²: {emissions_model.rsquared_adj:.4f}\n")
    output_file.write(f"  - F-statistic: {emissions_model.fvalue:.2f} (p-value: {emissions_model.f_pvalue:.2e})\n")
    
    output_file.write(f"\nWait Time Model:\n")
    output_file.write(f"  - R²: {wait_time_model.rsquared:.4f}\n")
    output_file.write(f"  - Adjusted R²: {wait_time_model.rsquared_adj:.4f}\n")
    output_file.write(f"  - F-statistic: {wait_time_model.fvalue:.2f} (p-value: {wait_time_model.f_pvalue:.2e})\n")
    
    output_file.write(f"\nTOP PREDICTORS:\n")
    output_file.write(f"Carbon Emissions (Top 5):\n")
    for i, row in feature_importance_emissions.head(5).iterrows():
        output_file.write(f"  - {row['feature']}: {row['coefficient']:.4e} (p={row['p_value']:.3e})\n")
    
    output_file.write(f"\nWait Time (Top 5):\n")
    for i, row in feature_importance_wait.head(5).iterrows():
        output_file.write(f"  - {row['feature']}: {row['coefficient']:.4e} (p={row['p_value']:.3e})\n")
    
    # Key insights
    output_file.write(f"\nKEY INSIGHTS:\n")
    
    # Carbon emissions insights
    significant_emissions = feature_importance_emissions[feature_importance_emissions['p_value'] < 0.05]
    output_file.write(f"- {len(significant_emissions)} features significantly predict carbon emissions\n")
    
    if 'power_x_runtime' in significant_emissions['feature'].values:
        output_file.write("- Power-runtime interaction is a significant predictor of emissions\n")
    
    if 'carbon_consideration' in significant_emissions['feature'].values:
        coef = significant_emissions[significant_emissions['feature'] == 'carbon_consideration']['coefficient'].iloc[0]
        if coef < 0:
            output_file.write("- Higher carbon consideration is associated with lower emissions\n")
        else:
            output_file.write("- Higher carbon consideration is associated with higher emissions\n")
    
    # Wait time insights
    significant_wait = feature_importance_wait[feature_importance_wait['p_value'] < 0.05]
    output_file.write(f"- {len(significant_wait)} features significantly predict wait time\n")
    
    if 'queue_length_at_submission' in significant_wait['feature'].values:
        output_file.write("- Queue length at submission significantly affects wait time\n")
    
    # Correlations
    corr_power_emissions = df['power'].corr(df['carbon_emissions'])
    corr_queue_wait = df['queue_length_at_submission'].corr(df['wait_time'])
    
    output_file.write(f"- Power and emissions correlation: {corr_power_emissions:.3f}\n")
    output_file.write(f"- Queue length and wait time correlation: {corr_queue_wait:.3f}\n")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='OLS Regression Analysis for Job Details')
    parser.add_argument('--experiment', required=True, help='Experiment folder path')
    parser.add_argument('--epoch', type=int, help='Epoch number to analyze (if not specified, uses final)')
    
    args = parser.parse_args()
    
    try:
        # Determine the path to job details CSV based on experiment and epoch
        if args.epoch is not None:
            validation_dir = f"{args.experiment}/validation_results/epoch_{args.epoch}"
            epoch_label = f"epoch_{args.epoch}"
        else:
            validation_dir = f"{args.experiment}/validation_results/final"
            epoch_label = "final"
        
        csv_file = f"{validation_dir}/job_details_enhanced.csv"
        
        # Check if the CSV file exists
        if not os.path.exists(csv_file):
            # List available validation results to help user
            validation_base = f"{args.experiment}/validation_results"
            if os.path.exists(validation_base):
                available_results = []
                for item in os.listdir(validation_base):
                    item_path = f"{validation_base}/{item}"
                    if os.path.isdir(item_path):
                        job_file = f"{item_path}/job_details_enhanced.csv"
                        if os.path.exists(job_file):
                            available_results.append(item)
                
                if available_results:
                    print(f"Available validation results in {validation_base}/:")
                    for result in sorted(available_results):
                        print(f"  - {result}")
                else:
                    print(f"No validation results found in {validation_base}/")
            
            raise FileNotFoundError(f"Job details CSV not found: {csv_file}")
        
        # Create experiment directory if it doesn't exist
        os.makedirs(args.experiment, exist_ok=True)
        
        # Generate output filename without timestamp
        output_filename = f"{args.experiment}/regression_analysis_{epoch_label}.txt"
        
        with open(output_filename, 'w') as output_file:
            output_file.write("="*80 + "\n")
            output_file.write("JOB REGRESSION ANALYSIS REPORT\n")
            output_file.write("="*80 + "\n")
            output_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output_file.write(f"Input file: {csv_file}\n")
            output_file.write(f"Experiment: {args.experiment}\n")
            output_file.write(f"Epoch: {epoch_label}\n")
            output_file.write("="*80 + "\n")
            
            # Load and prepare data
            df = load_and_prepare_data(csv_file, output_file)
            
            # Create feature matrix
            X, base_features, interaction_features = create_feature_matrix(df, output_file)
            feature_names = base_features + interaction_features
            
            # Perform regression analysis for carbon emissions
            emissions_model, feature_importance_emissions = perform_regression_analysis(
                X, df['carbon_emissions'], 'Carbon Emissions', feature_names, output_file
            )
            
            # Perform regression analysis for wait time
            wait_time_model, feature_importance_wait = perform_regression_analysis(
                X, df['wait_time'], 'Wait Time', feature_names, output_file
            )
            
            # Generate summary report
            generate_summary_report(df, emissions_model, wait_time_model, 
                                   feature_importance_emissions, feature_importance_wait, output_file)
            
            output_file.write(f"\n{'='*80}\n")
            output_file.write("ANALYSIS COMPLETED SUCCESSFULLY\n")
            output_file.write(f"{'='*80}\n")
        
        # Only print the final success message
        print(f"Analysis complete! Results saved to: {output_filename}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 