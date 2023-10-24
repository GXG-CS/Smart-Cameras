import subprocess

# Relative path to your data file
data_path = "../results/pi3b/tf_pose_estimation/cluster/kmeans/robust_3.csv"

# Relative output directory for regression results
output_dir = "../results/regression/kmeans_robust_3/"

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

# Loop through each pair of regression and preprocessing methods
for reg_method, preproc_methods in method_dict.items():
    for preproc_method in preproc_methods:
        # Construct the command to run regression.py
        cmd = f"python ../regression.py --data {data_path} --method {reg_method} --preprocessing {preproc_method} --output {output_dir}"

        # Run the command
        subprocess.run(cmd, shell=True)

    # If no preprocessing is defined for the regression method, run without preprocessing
    if not preproc_methods:
        cmd = f"python ../regression.py --data {data_path} --method {reg_method} --output {output_dir}"
        subprocess.run(cmd, shell=True)
