# GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning

GAS-MARL is a green-aware job scheduling algorithm for HPC clusters based on multi-action deep reinforcement learning, which optimizes both renewable energy utilization and average bounded slowdown. This repository contains the source code of GAS-MARL and the datasets used.

## Install

All necessary packages can be installed with

```
pip install -r requirements.txt
```

## Configuration

The system uses a centralized configuration file `configFile/config.ini` that contains all parameters and constants. This makes it easy to manage different experimental setups and ensures reproducibility.

### Configuration File Structure

```ini
[GAS-MARL setting]
eta = 0.00
MAX_QUEUE_SIZE = 256
run_win = 64
green_win = 200
delayMaxJobNum=5
delayTimeList=[900,1800,3600,7200,22100,44200,86400]

[general setting]
processor_per_machine = 8
idlePower = 50
MAX_perProcPower=50
carbon_year = 2021

[carbon setting]
USE_DYNAMIC_WINDOW = True

[algorithm constants]
MAX_POWER = 19000
MAX_GREEN = 19000
MAX_WAIT_TIME = 43200
MAX_RUN_TIME = 43200
JOB_FEATURES = 7
JOB_SEQUENCE_SIZE = 64
RUN_FEATURE = 4
GREEN_FEATURE = 2
MAX_CARBON_INTENSITY = 500.0

[training parameters]
seed = 0
epochs = 300
traj_num = 100
```

### Parameter Descriptions

#### [GAS-MARL setting]
- **`eta`**: Penalty factors for model training
- **`MAX_QUEUE_SIZE`**: Maximum number of jobs in the waiting queue
- **`run_win`**: Maximum number of running jobs tracked
- **`green_win`**: Number of time slots in renewable energy information
- **`delayMaxJobNum`**: Maximum number of jobs that can be delayed
- **`delayTimeList`**: Available delay times in seconds

#### [general setting]
- **`processor_per_machine`**: Number of processors per machine
- **`idlePower`**: Idle power consumption per machine (watts)
- **`MAX_perProcPower`**: Maximum power per processor (watts)
- **`carbon_year`**: Year for carbon intensity data (2021-2024)

#### [carbon setting]
- **`USE_DYNAMIC_WINDOW`**: If True, use dynamic carbon window; if False, use fixed 24-hour window

#### [algorithm constants]
- **`MAX_POWER`**: Maximum power value for normalization
- **`MAX_GREEN`**: Maximum green energy value
- **`MAX_WAIT_TIME`**: Maximum job wait time (seconds)
- **`MAX_RUN_TIME`**: Maximum job runtime (seconds)
- **`JOB_FEATURES`**: Number of features per job (7)
- **`JOB_SEQUENCE_SIZE`**: Job sequence size for episodes
- **`RUN_FEATURE`**: Features for running jobs
- **`GREEN_FEATURE`**: Features for green energy
- **`MAX_CARBON_INTENSITY`**: Maximum carbon intensity for normalization

#### [training parameters]
- **`seed`**: Random seed for reproducibility
- **`epochs`**: Number of training epochs
- **`traj_num`**: Number of trajectories per epoch

## Training

### Enhanced Training Interface

The training system now supports organized experiments with detailed tracking:

```bash
python MARL.py --workload [str] --backfill [int] --name [str] --description [str]
```

**Required Arguments:**
- **`--workload`**: Job trace name (lublin_256, Cirne, Jann)
- **`--name`**: Experiment name (e.g., ED12, baseline, carbon_opt)

**Optional Arguments:**
- **`--backfill`**: Backfill policy (0=None, 1=Green-backfilling, 2=EASY-backfilling)
- **`--description`**: Description of the experiment
- **`--debug`**: Enable debug output

### Experiment Directory Structure

Each experiment creates an organized directory structure:

```
workload_name/
└── MARL_experiment_name/
    ├── description.txt              # Experiment metadata
    ├── config_snapshot.ini          # Configuration used
    ├── training_results.csv         # Training progress
    ├── checkpoints/                 # Model checkpoints
    │   ├── epoch_5/
    │   ├── epoch_10/
    │   └── ...
    └── final/                       # Final trained model
        ├── _actor.pkl
        └── _critic.pkl
```

### Training Examples

**Basic training:**
```bash
python MARL.py --workload lublin_256 --backfill 0 --name baseline --description "Baseline MARL without backfilling"
```

**Carbon-optimized experiment:**
```bash
python MARL.py --workload lublin_256 --backfill 1 --name carbon_v1 --description "Carbon-aware scheduling with green backfilling"
```

**Debug mode:**
```bash
python MARL.py --workload lublin_256 --backfill 0 --name debug_test --description "Debug run" --debug
```

## Validation

### Enhanced Model Validation

Use the `validate.py` script to perform detailed validation of trained models with comprehensive job-level tracking and epoch-specific analysis:

```bash
python validate.py --experiment [str] --workload [str] --epoch [int] --episodes [int] --backfill [int]
```

**Required Arguments:**
- **`--experiment`**: Experiment name (e.g., "carbon_v1") or full path (e.g., "lublin_256_carbon_float/MARL_carbon_v1")
- **`--workload`**: Workload name (e.g., "lublin_256_carbon_float") or file path (e.g., "./data/lublin_256_carbon_float.swf")

**Optional Arguments:**
- **`--epoch`**: Epoch number to validate (e.g., 25, 50). If not specified, validates final weights
- **`--episodes`**: Number of validation episodes (default: 5)
- **`--backfill`**: Backfill strategy (0=FCFS, 1=backfill enabled, default: 0)
- **`--seed`**: Random seed for reproducibility (default: 42)
- **`--debug`**: Enable debug output

### Epoch-Specific Validation

The validation system now supports epoch-specific analysis, allowing you to validate models at different training stages:

**Validate specific epoch:**
```bash
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --epoch 25
```

**Validate final weights:**
```bash
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float
```

### Validation Features

The enhanced validation script provides:

- **Epoch-Specific Analysis**: Validate models from any training checkpoint
- **Organized Output**: Results saved in epoch-specific folders
- **Detailed Job Tracking**: Records every job's submission, scheduling, and completion
- **Carbon Analysis**: Tracks carbon consideration levels (0.0-1.0) and their distribution
- **Performance Metrics**: Wait times, completion times, queue dynamics
- **Statistical Analysis**: Mean, std, and distribution analysis
- **Clean CSV Output**: Organized results without timestamps

### Validation Examples

**Validate specific training epochs:**
```bash
# Validate early training (epoch 10)
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --epoch 10 --episodes 10

# Validate mid-training (epoch 25)
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --epoch 25 --episodes 10

# Validate late training (epoch 50)
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --epoch 50 --episodes 10
```

**Validate final model:**
```bash
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --episodes 20
```

**Compare different experiments:**
```bash
python validate.py --experiment baseline --workload lublin_256_carbon_float --episodes 10
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --episodes 10
```

### Validation Output Structure

The script generates organized output in epoch-specific directories:

```
experiment_folder/
└── validation_results/
    ├── epoch_10/
    │   ├── job_details_enhanced.csv      # Individual job records
    │   └── episode_summary_enhanced.csv  # Episode-level summaries
    ├── epoch_25/
    │   ├── job_details_enhanced.csv
    │   └── episode_summary_enhanced.csv
    └── final/
        ├── job_details_enhanced.csv
        └── episode_summary_enhanced.csv
```

### Validation Output Analysis

Example console output with comprehensive statistics:
```
Enhanced Job Statistics:
  Total jobs tracked: 1024
  Jobs scheduled: 1019
  Jobs completed: 1015
  Average wait time: 1245.67s
  
  Carbon consideration stats:
    Min: 0.034, Max: 0.987, Mean: 0.512, Std: 0.289
    0.0-0.2: 89 jobs (8.7%)
    0.2-0.4: 156 jobs (15.3%)
    0.4-0.6: 201 jobs (19.7%)
    0.6-0.8: 178 jobs (17.5%)
    0.8-1.0: 395 jobs (38.8%)
  
  Carbon emissions stats:
    Total emissions: 1234.56 gCO2eq
    Mean per job: 1.22 gCO2eq
    Std: 0.45 gCO2eq
```

## Regression Analysis

### Job Performance Analysis

Use the `analyze_job_regression.py` script to perform comprehensive OLS regression analysis on validation results:

```bash
python analyze_job_regression.py --experiment [str] --epoch [int]
```

**Required Arguments:**
- **`--experiment`**: Experiment folder path (e.g., "lublin_256_carbon_float/MARL_carbon_v1")

**Optional Arguments:**
- **`--epoch`**: Epoch number to analyze. If not specified, analyzes final weights results

### Regression Analysis Features

The analysis script provides:

- **Automatic Data Location**: Finds job details CSV based on experiment and epoch
- **Comprehensive Feature Engineering**: Creates base features and meaningful interactions
- **Dual Target Analysis**: Analyzes both carbon emissions and wait times
- **Statistical Modeling**: OLS regression with significance testing
- **Feature Importance**: Ranks predictors by impact and significance
- **Domain Insights**: Interprets results in scheduling context
- **Clean TXT Output**: All results saved to experiment folder

### Analysis Examples

**Analyze specific epoch:**
```bash
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1 --epoch 25
```

**Analyze final model:**
```bash
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1
```

**Compare multiple epochs:**
```bash
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1 --epoch 10
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1 --epoch 25
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1 --epoch 50
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1
```

### Analysis Output

The script generates comprehensive TXT reports:

```
experiment_folder/
├── regression_analysis_epoch_10.txt    # Epoch 10 analysis
├── regression_analysis_epoch_25.txt    # Epoch 25 analysis
├── regression_analysis_epoch_50.txt    # Epoch 50 analysis
└── regression_analysis_final.txt       # Final model analysis
```

### Features Analyzed

**Base Features:**
- `request_time`: Job runtime request
- `request_processors`: Number of processors requested
- `carbon_consideration`: Carbon awareness factor (0-1)
- `queue_length_at_submission`: Queue size when submitted
- `power`: Power consumption (watts)
- `wait_time`: Time waited in queue

**Interaction Features:**
- Power-related: `power_x_runtime`, `power_x_processors`
- Queue dynamics: `queue_x_processors`, `queue_x_runtime`
- Carbon interactions: `carbon_x_power`, `carbon_x_runtime`, `carbon_x_queue`
- Resource utilization: `processors_x_runtime`, `wait_x_runtime`, `wait_x_processors`
- Efficiency ratios: `log_power_per_processor`, `log_wait_per_queue`

**Target Variables:**
- `carbon_emissions`: Actual carbon emissions (gCO2eq)
- `wait_time`: Queue waiting time (seconds)

## Testing (Legacy Comparison)

For comparing multiple algorithms, use the legacy comparison script:

```bash
python compare.py --workload [str] --len [int] --iter [int] --backfill [int]
```

**Arguments:**
- **`--workload`**: Job trace name
- **`--len`**: Length of scheduling sequence
- **`--iter`**: Number of sequences sampled
- **`--backfill`**: Backfill policy

## Complete Workflow Example

Here's a complete example of training, validating, and analyzing an experiment:

### 1. Configure Parameters
Edit `configFile/config.ini` with your desired settings.

### 2. Train Model
```bash
python MARL.py --workload lublin_256_carbon_float --backfill 1 --name carbon_v1 --description "Carbon-aware MARL with green backfilling and dynamic window"
```

### 3. Monitor Training
Training progress is saved to `lublin_256_carbon_float/MARL_carbon_v1/training_results.csv`

### 4. Validate at Different Training Stages
```bash
# Validate early training checkpoint
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --epoch 10 --episodes 10

# Validate mid-training checkpoint  
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --epoch 25 --episodes 10

# Validate final model
python validate.py --experiment carbon_v1 --workload lublin_256_carbon_float --episodes 20
```

### 5. Perform Regression Analysis
```bash
# Analyze different training stages
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1 --epoch 10
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1 --epoch 25
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_carbon_v1

# Compare with baseline
python validate.py --experiment baseline --workload lublin_256_carbon_float --episodes 10
python analyze_job_regression.py --experiment lublin_256_carbon_float/MARL_baseline
```

### 6. Review Results
Check the organized output files:
```
lublin_256_carbon_float/MARL_carbon_v1/
├── validation_results/
│   ├── epoch_10/
│   │   ├── job_details_enhanced.csv
│   │   └── episode_summary_enhanced.csv
│   ├── epoch_25/
│   │   ├── job_details_enhanced.csv
│   │   └── episode_summary_enhanced.csv
│   └── final/
│       ├── job_details_enhanced.csv
│       └── episode_summary_enhanced.csv
├── regression_analysis_epoch_10.txt
├── regression_analysis_epoch_25.txt
└── regression_analysis_final.txt
```

## File Structure

```
green-rl-sched/
├── configFile/
│   └── config.ini                    # Centralized configuration
├── data/                             # Workload files and carbon data
├── MARL.py                           # Main training script
├── validate.py                       # Enhanced validation script
├── analyze_job_regression.py         # Regression analysis script
├── compare.py                        # Multi-algorithm comparison
├── HPCSimPickJobs.py                # HPC environment simulation
├── greenPower.py                    # Carbon-aware components
└── workload_name/                   # Experiment results
    └── MARL_experiment_name/         # Individual experiments
        ├── description.txt
        ├── config_snapshot.ini
        ├── training_results.csv
        ├── checkpoints/              # Training checkpoints
        │   ├── epoch_5/
        │   ├── epoch_10/
        │   └── ...
        ├── final/                    # Final trained model
        ├── validation_results/       # Validation outputs
        │   ├── epoch_10/
        │   │   ├── job_details_enhanced.csv
        │   │   └── episode_summary_enhanced.csv
        │   ├── epoch_25/
        │   │   ├── job_details_enhanced.csv
        │   │   └── episode_summary_enhanced.csv
        │   └── final/
        │       ├── job_details_enhanced.csv
        │       └── episode_summary_enhanced.csv
        ├── regression_analysis_epoch_10.txt    # Regression analysis reports
        ├── regression_analysis_epoch_25.txt
        └── regression_analysis_final.txt
```

## Key Features

- **🔧 Centralized Configuration**: All parameters in `config.ini`
- **📁 Organized Experiments**: Structured directories with metadata
- **📊 Enhanced Validation**: Epoch-specific job-level tracking and analysis
- **📈 Regression Analysis**: Comprehensive statistical modeling of scheduling patterns
- **🌱 Carbon-Aware**: Carbon consideration tracking and optimization
- **🔄 Reproducible**: Saved configurations and fixed seeds
- **📋 Progress Tracking**: Real-time training and validation monitoring
- **🎯 Epoch Comparison**: Easy comparison across training stages

## Citing GAS-MARL

```
If you reference or use GAS-MARL in your research, please cite:
@article{CHEN2025107760,
title = {GAS-MARL: Green-Aware job Scheduling algorithm for HPC clusters based on Multi-Action Deep Reinforcement Learning},
journal = {Future Generation Computer Systems},
volume = {167},
pages = {107760},
year = {2025},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2025.107760},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X2500055X},
author = {Rui Chen and Weiwei Lin and Huikang Huang and Xiaoying Ye and Zhiping Peng},
keywords = {Job scheduling, High-performance computing, Deep Reinforcement Learning, Renewable energy, Green computing},
}
```

### Acknowledgment

We extend our heartfelt appreciation to the following GitHub repositories for providing valuable code bases:

https://github.com/DIR-LAB/deep-batch-scheduler

http://github.com/laurentphilippe/greenpower
