import os
import psutil
import time
import resource
from gpiozero import CPUTemperature
from sklearn.datasets import make_regression
import lightgbm as lgb
import argparse

def set_cpu_speed(ghz):
    mhz = ghz * 1000  # Convert GHz to MHz
    os.system(f'sudo cpufreq-set -f {mhz}MHz')

def limit_cpu_cores_for_process(process_id, num_cores):
    os.system(f'taskset -pc 0-{num_cores-1} {process_id}')

def limit_memory_usage(gb_limit):
    """
    Limit memory usage for the process in GB using the resource module.
    Convert the limit from GB to MB for setting the memory limit.
    """
    mb_limit = gb_limit * 1024  # Convert GB to MB
    # Convert MB to Bytes for setrlimit
    byte_limit = int(mb_limit * 1024**2)
    resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))

def get_process_memory_usage():
    """
    Get the memory usage of the current process in GB.
    """
    process = psutil.Process(os.getpid())
    mem_in_bytes = process.memory_info().rss
    mem_in_gb = mem_in_bytes / (1024**3)
    return mem_in_gb

def main(cpu_cores, cpu_freq, mem):
    # Set CPU parameters
    current_process_id = os.getpid()
    limit_cpu_cores_for_process(current_process_id, cpu_cores)
    set_cpu_speed(cpu_freq)
    limit_memory_usage(mem)

    # Log system and process info
    p = psutil.Process(current_process_id)
    actual_cpu_cores = p.cpu_affinity()
    print(f"Storage Capacity (GB): {psutil.disk_usage('/').total / (1024**3)}")
    print(f"RAM Capacity (GB): {psutil.virtual_memory().total / (1024**3)}")
    print(f"RAM Used by Process (GB): {get_process_memory_usage()}")
    print(f"CPU Speed (GHz): {psutil.cpu_freq().current / 1000}")
    print(f"Number of CPU cores process is running on: {len(actual_cpu_cores)}")
    print(f"CPU Temperature (°C): {CPUTemperature().temperature}")

    # Training a LightGBM model
    print("Beginning model training...")
    start_time = time.time()
    X, y = make_regression(n_samples=10000, n_features=200, noise=0.1)
    dataset = lgb.Dataset(X, label=y)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    num_round = 100
    lgb.train(params, dataset, num_round)
    end_time = time.time()
    print("Model training completed!")
    print(f"Training Time: {end_time - start_time} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu_cores', type=int, required=True, help='Number of CPU cores to use.')
    parser.add_argument('--cpu_freq', type=float, required=True, help='Frequency of CPU in GHz.')
    parser.add_argument('--mem', type=float, required=True, help='Memory limit for the process in GB.')
    args = parser.parse_args()

    main(args.cpu_cores, args.cpu_freq, args.mem)

