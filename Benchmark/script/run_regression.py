import subprocess
import os

# Relative output directory for regression results
output_dir = "../results/pi2b/tf_pose_estimation/regression/gaussian_mixture_yeojohnson_4/"

# Check if the output directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Relative path to your data file
# data_path = "../results/pi2b/tf_pose_estimation/cluster/kmeans/robust_3.csv"
data_path = "../data_cleaning/pi2b/pi2b_yeojohnson_4_filtered.csv"

# Dictionary of regression_method: [list of preprocessing_methods]
method_dict = {
    'linear': ['standard', 'minmax'],
    'ridge': ['standard'],
    'lasso': ['robust', 'maxabs'],
    'decision_tree': [],
    'random_forest': ['quantile'],
    'svr': ['yeojohnson', 'log'],
    'knn': ['standard', 'minmax'],
    'gbr': ['robust'],
    'neural_network': ['binarizer']
}

# python regression.py --data results/pi3b/tf_pose_estimation/cluster/kmeans/robust_3.csv --method svr --preprocessing standard --output results/pi3b/tf_pose_estimation/regression/kmeans_robust_3


# Loop through each pair of regression and preprocessing methods
for reg_method, preproc_methods in method_dict.items():
    for preproc_method in preproc_methods:
        # Construct the command to run regression.py
        cmd = f"python ../regression_new.py --data {data_path} --method {reg_method} --preprocessing {preproc_method} --output {output_dir}"

        # Log the command
        print(f"Running command: {cmd}")

        # Run the command
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to run {cmd}")

    # If no preprocessing is defined for the regression method, run without preprocessing
    if not preproc_methods:
        cmd = f"python ../regression.py --data {data_path} --method {reg_method} --output {output_dir} "
        
        # Log the command
        print(f"Running command: {cmd}")

        # Run the command
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to run {cmd}")

