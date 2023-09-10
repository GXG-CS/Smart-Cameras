import os
import psutil
import time
from gpiozero import CPUTemperature
from sklearn.datasets import make_regression
import lightgbm as lgb
import argparse

def set_cpu_speed(ghz):
    """
    Set CPU speed in GHz.
    Requires sudo privilege.
    """
    mhz = ghz * 1000  # Convert GHz to MHz
    os.system(f'sudo cpufreq-set -f {mhz}MHz')

def limit_cpu_cores_for_process(process_id, num_cores):
    """
    Limit the process to run on the specified number of CPU cores.
    """
    os.system(f'taskset -pc 0-{num_cores-1} {process_id}')

def main(cpu_cores, cpu_freq):
    # Set CPU parameters
    current_process_id = os.getpid()
    limit_cpu_cores_for_process(current_process_id, cpu_cores)
    set_cpu_speed(cpu_freq)
    
    # Getting the actual CPU cores the process is allowed to use
    p = psutil.Process(current_process_id)
    actual_cpu_cores = p.cpu_affinity()

    # Data collection
    total, used, free = map(int, os.popen('df -Pk / | tail -1').read().split()[1:4])
    Scap = total / 10**6  # Convert KB to GB
    Rcap = psutil.virtual_memory().total / (1024**3)  # Convert Bytes to GB
    Rused = psutil.virtual_memory().used / (1024**3)
    Pspeed = psutil.cpu_freq().current / 1000  # Convert MHz to GHz
    Pcores = len(actual_cpu_cores) # using the length of the list of actual CPU cores
    cpu_temp = CPUTemperature().temperature

    print(f"Storage Capacity (GB): {Scap}")
    print(f"RAM Capacity (GB): {Rcap}")
    print(f"RAM Used (GB): {Rused}")
    print(f"CPU Speed (GHz): {Pspeed}")
    print(f"Number of CPU cores process is running on: {Pcores}")
    print(f"CPU Temperature (°C): {cpu_temp}")

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
    args = parser.parse_args()
    
    main(args.cpu_cores, args.cpu_freq)

