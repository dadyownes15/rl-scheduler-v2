import configparser
from train import PPO as CCScheduler
import torch
import os
from HPCSimPickJobs import HPCEnv

def load_settings(experiment_path):
    config = configparser.ConfigParser()
    config_path = experiment_path + "/config_snapshot.ini"
    config.read(config_path)


    return {
    'eta': float(config.get('GAS-MARL setting', 'eta')),
    'MAX_QUEUE_SIZE': int(config.get('GAS-MARL setting', 'MAX_QUEUE_SIZE')),
    'run_win': int(config.get('GAS-MARL setting', 'run_win')),
    'green_win': int(config.get('GAS-MARL setting', 'green_win')),
    'delayMaxJobNum': int(config.get('GAS-MARL setting', 'delayMaxJobNum')),
    'delayTimeList': eval(config.get('GAS-MARL setting', 'delayTimeList')),

    'processor_per_machine': int(config.get('general setting', 'processor_per_machine')),
    'idlePower': float(config.get('general setting', 'idlePower')),
    'MAX_perProcPower': float(config.get('general setting', 'MAX_perProcPower')),
    'carbon_year': int(config.get('general setting', 'carbon_year')),

    'MAX_POWER': int(config.get('algorithm constants', 'MAX_POWER')),
    'MAX_GREEN': int(config.get('algorithm constants', 'MAX_GREEN')),
    'MAX_WAIT_TIME': int(config.get('algorithm constants', 'MAX_WAIT_TIME')),
    'MAX_RUN_TIME': int(config.get('algorithm constants', 'MAX_RUN_TIME')),
    'JOB_FEATURES': int(config.get('algorithm constants', 'JOB_FEATURES')),
    'JOB_SEQUENCE_SIZE': int(config.get('algorithm constants', 'JOB_SEQUENCE_SIZE')),
    'RUN_FEATURE': int(config.get('algorithm constants', 'RUN_FEATURE')),
    'GREEN_FEATURE': int(config.get('algorithm constants', 'GREEN_FEATURE')),
    'BASE_LINE_WAIT_CARBON_PENALITY': float(config.get('algorithm constants', 'BASE_LINE_WAIT_CARBON_PENALITY'))
    }

def load_ccscheduler_model(experiment_path, workload_arg=None, epoch=None, ):
    """
    Load CCScheduler model from experiment directory.
    
    Args:
        experiment_path: Experiment name (e.g., "MARL_basic", "basic") or full path
        workload_arg: Workload name or file path (e.g., "lublin_256_carbon_float" or "./data/lublin_256_carbon_float.swf")
        epoch: Epoch number to load weights from (if None, loads from final/)
    
    Returns:
        Loaded CCScheduler model
    """

    config = load_settings(experiment_path=experiment_path)

    # Derive green_rows = ceil((green_win + 8) / JOB_FEATURES) for unified green vector (forecast+context)
    green_rows = (config['green_win'] + 8 + config['JOB_FEATURES'] - 1) // config['JOB_FEATURES']
    inputNum_size = [config['MAX_QUEUE_SIZE'], config['run_win'], green_rows]
    featureNum_size = [config['JOB_FEATURES'], config['RUN_FEATURE'], config['GREEN_FEATURE']]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device: ", device)
    
    model = CCScheduler(batch_size=256, inputNum_size=inputNum_size,
                 featureNum_size=featureNum_size, device=device)
    
    # Auto-detect experiment path if only experiment name is provided
    if '/' not in experiment_path and workload_arg:
        # Extract workload name from argument
        if workload_arg.endswith('.swf'):
            # It's a file path like "./data/lublin_256_carbon_float.swf"
            workload_name = os.path.basename(workload_arg).replace('.swf', '')
        else:
            # It's just a workload name like "lublin_256_carbon_float"
            workload_name = workload_arg
        
        # Handle experiment name - check if it already has MARL_ prefix
        if experiment_path.startswith('MARL_'):
            # Already has prefix, use as-is
            full_experiment_path = f"{workload_name}/{experiment_path}"
        else:
            # Add MARL_ prefix
            full_experiment_path = f"{workload_name}/MARL_{experiment_path}"
        
        if os.path.exists(full_experiment_path):
            experiment_path = full_experiment_path
            print(f"Auto-detected experiment path: {experiment_path}")
        else:
            print(f"Warning: Could not auto-detect path for experiment '{experiment_path}', using as provided")
    
    # Determine weights path based on epoch
    if epoch is not None:
        # Load from specific epoch checkpoint
        weights_path = f"{experiment_path}/checkpoints/epoch_{epoch}/"
        if not os.path.exists(f"{weights_path}_actor.pkl"):
            # List available epochs to help user
            checkpoints_dir = f"{experiment_path}/checkpoints"
            if os.path.exists(checkpoints_dir):
                available_epochs = []
                for item in os.listdir(checkpoints_dir):
                    if item.startswith('epoch_') and os.path.isdir(f"{checkpoints_dir}/{item}"):
                        epoch_num = item.replace('epoch_', '')
                        if os.path.exists(f"{checkpoints_dir}/{item}/_actor.pkl"):
                            available_epochs.append(epoch_num)
                available_epochs.sort(key=int)
                print(f"Available epochs in {checkpoints_dir}/:")
                for epoch_num in available_epochs:
                    print(f"  - epoch_{epoch_num}")
            raise FileNotFoundError(f"No trained weights found for epoch {epoch} in {experiment_path}/checkpoints/")
    else:
        # Load from final weights, fallback to legacy location
        weights_path = f"{experiment_path}/final/"
        if not os.path.exists(f"{weights_path}_actor.pkl"):
            # Try legacy location
            workload = experiment_path.split('/')[0]
            weights_path = f"{workload}/MARL/"
            if not os.path.exists(f"{weights_path}_actor.pkl"):
                # List available experiments to help user
                if workload_arg:
                    # Extract workload name for listing
                    if workload_arg.endswith('.swf'):
                        workload_name = os.path.basename(workload_arg).replace('.swf', '')
                    else:
                        workload_name = workload_arg
                        
                    print(f"Available experiments in {workload_name}/:")
                    if os.path.exists(workload_name):
                        experiments = [d for d in os.listdir(workload_name) if d.startswith('MARL_')]
                        for exp in experiments:
                            print(f"  - {exp}")
                raise FileNotFoundError(f"No trained weights found in {experiment_path}")
    
    print(f"Loading weights from: {weights_path}")
    model.load_using_model_name(weights_path)
    
    # Store the final experiment path for later use
    model._experiment_path = experiment_path
    
    return model



def initialize_model(experiment_path, workload, epoch):
    # Load model
    exp_for_load = experiment_path 
    model = load_ccscheduler_model(exp_for_load, workload, epoch)
    model_device = next(model.actor_net.parameters()).device if hasattr(model, 'actor_net') else torch.device('cpu')
    print("Model loaded on:", model_device)
    return model

def initialize_environment(workload, backfill, debug, seed, jobs_per_episode):
    # Resolve workload path
    if workload.endswith('.swf'):
        workload_file = workload
    else:
        workload_file = f"./data/{workload}.swf"
    print("Using workload:", workload_file)

    # Initialize environment
    env = HPCEnv(backfill=backfill, debug=debug)
    env.my_init(workload_file)
    env.seed(seed)

    # Override job count for the episode
    env._validation_job_sequence_size = jobs_per_episode

    # Prepare episode with exactly JOBS_PER_EPISODE jobs
    env.reset()
    if hasattr(env, '_validation_job_sequence_size'):
        # Bound within available jobs
        max_jobs_available = len(env.loads.all_jobs) - env.start
        actual_job_count = min(env._validation_job_sequence_size, max_jobs_available)
        env.num_job_in_batch = actual_job_count
        env.last_job_in_batch = env.start + env.num_job_in_batch

    return env
