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

### Detailed Model Validation

Use the `validate.py` script to perform detailed validation of trained models with comprehensive job-level tracking:

```bash
python validate.py --experiment [str] --episodes [int] --backfill [int]
```

**Arguments:**
- **`--experiment`**: Experiment name to validate (e.g., ED12)
- **`--episodes`**: Number of validation episodes (default: 5)
- **`--workload`**: Workload dataset (default: lublin_256)
- **`--backfill`**: Backfill strategy (default: 0)
- **`--sequence_length`**: Length of job sequence to evaluate (default: 1024)
- **`--debug`**: Enable debug output

### Validation Features

The validation script provides:

- **Detailed Job Tracking**: Records every job's submission, scheduling, and completion
- **Carbon Analysis**: Tracks carbon consideration levels (0.0-1.0) and their distribution
- **Performance Metrics**: Wait times, completion times, queue dynamics
- **Statistical Analysis**: Mean, std, and distribution analysis
- **CSV Output**: Detailed results saved to experiment directory

### Validation Examples

**Basic validation:**
```bash
python validate.py --experiment baseline --episodes 10
```

**Extended validation:**
```bash
python validate.py --experiment carbon_v1 --episodes 20 --backfill 1 --sequence_length 2048
```

**Quick debug validation:**
```bash
python validate.py --experiment debug_test --episodes 1 --debug --sequence_length 100
```

### Validation Output

The script generates:

1. **Console Output**: Real-time progress and summary statistics
2. **CSV Files**: 
   - `job_details_[timestamp].csv`: Individual job records
   - `episode_summary_[timestamp].csv`: Episode-level summaries
3. **Carbon Analysis**: Distribution of carbon consideration levels
4. **Performance Stats**: Wait times, completion rates, reward metrics

Example output:
```
Job Statistics:
  Total jobs tracked: 1024
  Jobs scheduled: 1019
  Average wait time: 1245.67s
  Carbon consideration distribution:
    Min: 0.034
    Max: 0.987
    Mean: 0.512
    Std: 0.289
    0.0-0.2: 89 jobs (8.7%)
    0.2-0.4: 156 jobs (15.3%)
    0.4-0.6: 201 jobs (19.7%)
    0.6-0.8: 178 jobs (17.5%)
    0.8-1.0: 395 jobs (38.8%)
```

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

Here's a complete example of training and validating an experiment:

### 1. Configure Parameters
Edit `configFile/config.ini` with your desired settings.

### 2. Train Model
```bash
python MARL.py --workload lublin_256 --backfill 1 --name carbon_exp_v1 --description "Carbon-aware MARL with green backfilling and dynamic window"
```

### 3. Monitor Training
Training progress is saved to `lublin_256/MARL_carbon_exp_v1/training_results.csv`

### 4. Validate Results
```bash
python validate.py --experiment carbon_exp_v1 --episodes 20 --backfill 1
```

### 5. Analyze Results
Check the validation output files in `lublin_256/MARL_carbon_exp_v1/validation_results/`

## File Structure

```
green-rl-sched/
├── configFile/
│   └── config.ini              # Centralized configuration
├── data/                       # Workload files and carbon data
├── MARL.py                     # Main training script
├── validate.py                 # Validation script
├── compare.py                  # Multi-algorithm comparison
├── HPCSimPickJobs.py          # HPC environment simulation
├── greenPower.py              # Carbon-aware components
└── workload_name/             # Experiment results
    └── MARL_experiment_name/   # Individual experiments
        ├── description.txt
        ├── config_snapshot.ini
        ├── training_results.csv
        ├── checkpoints/
        ├── final/
        └── validation_results/
```

## Key Features

- **🔧 Centralized Configuration**: All parameters in `config.ini`
- **📁 Organized Experiments**: Structured directories with metadata
- **📊 Detailed Validation**: Job-level tracking and analysis
- **🌱 Carbon-Aware**: Carbon consideration tracking and optimization
- **🔄 Reproducible**: Saved configurations and fixed seeds
- **📈 Progress Tracking**: Real-time training and validation monitoring

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
