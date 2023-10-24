import os
import subprocess

# Directory where the clustering results are located (with subdirectories like "kmeans", "optics", etc.)
DATA_DIR = '../results/pi2b/tf_pose_estimation/cluster'
# Base directory to save the evaluations
OUTPUT_BASE_DIR = '../results/pi2b/tf_pose_estimation/cluster_eval'

def main():
    # Iterate over subdirectories in DATA_DIR
    for method in os.listdir(DATA_DIR):
        method_dir = os.path.join(DATA_DIR, method)
        if os.path.isdir(method_dir):
            # Create corresponding output directory for the method
            method_output_dir = os.path.join(OUTPUT_BASE_DIR, method)
            if not os.path.exists(method_output_dir):
                os.makedirs(method_output_dir)

            # Iterate over .csv files in the method directory
            for filename in os.listdir(method_dir):
                if filename.endswith('.csv'):
                    full_path = os.path.join(method_dir, filename)
                    # Run the evaluation.py script for the current .csv file
                    cmd = ['python', '../evaluation.py', '--data', full_path, '--output', method_output_dir]
                    subprocess.run(cmd)

if __name__ == "__main__":
    main()
