import os
import subprocess
import psutil
import resource
import time
from gpiozero import CPUTemperature
from sklearn.datasets import make_regression
import lightgbm as lgb
import argparse

# --- Functions for Storage Limitation --- #
def create_limited_storage(image_path, mount_point, size_in_mb):
    subprocess.run(['dd', 'if=/dev/zero', f'of={image_path}', 'bs=1M', f'count={int(size_in_mb)}'])
    subprocess.run(['sudo', 'mkfs.ext4', image_path])
    os.makedirs(mount_point, exist_ok=True)
    subprocess.run(['sudo', 'mount', '-o', 'loop', image_path, mount_point])
    print(f"{image_path} mounted on {mount_point} with size {size_in_mb}MB")

def unmount_storage(mount_point):
    subprocess.run(['sudo', 'umount', mount_point])

# --- Functions for Memory and CPU Limitation --- #
def limit_memory_usage(gb_limit):
    mb_limit = gb_limit * 1024
    byte_limit = int(mb_limit * 1024**2)
    resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))

def set_cpu_speed(ghz):
    mhz = ghz * 1000
    os.system(f'sudo cpufreq-set -f {mhz}MHz')

def limit_cpu_cores_for_process(process_id, num_cores):
    os.system(f'taskset -pc 0-{num_cores-1} {process_id}')

def main(cpu_cores, cpu_freq, mem, storage):
    # Set limitations
    current_process_id = os.getpid()
    limit_cpu_cores_for_process(current_process_id, cpu_cores)
    set_cpu_speed(cpu_freq)
    limit_memory_usage(mem)
    image_path = "limited_storage.img"
    mount_point = "limited_storage_mount_point"
    create_limited_storage(image_path, mount_point, storage * 1024)  # Convert GB to MB for storage
    
    # Data collection and model training
    total, used, free = map(int, os.popen('df -Pk / | tail -1').read().split()[1:4])
    Scap = total / 10**6  # Convert KB to GB
    Rcap = psutil.virtual_memory().total / (1024**3)
    Rused = psutil.virtual_memory().used / (1024**3)
    Pspeed = psutil.cpu_freq().current / 1000  # Convert MHz to GHz
    Pcores = os.cpu_count()
    cpu_temp = CPUTemperature().temperature

    print(f"Storage Capacity (GB): {Scap}")
    print(f"RAM Capacity (GB): {Rcap}")
    print(f"RAM Used (GB): {Rused}")
    print(f"CPU Speed (GHz): {Pspeed}")
    print(f"Number of CPU cores: {Pcores}")
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
    
    # Unmount the limited storage
    unmount_storage(mount_point)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu_cores', type=int, required=True, help='Number of CPU cores to use.')
    parser.add_argument('--cpu_freq', type=float, required=True, help='Frequency of CPU in GHz.')
    parser.add_argument('--mem', type=float, required=True, help='Memory limit in GB.')
    parser.add_argument('--storage', type=float, required=True, help='Storage limit in GB.')
    args = parser.parse_args()
    
    main(args.cpu_cores, args.cpu_freq, args.mem, args.storage)

