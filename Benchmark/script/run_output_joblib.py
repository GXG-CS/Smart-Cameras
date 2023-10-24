import os
import subprocess

def run_for_all_models(directory):
    """
    Executes output_joblib.py for all .joblib files in the given directory.
    """

    # Create output directory if it doesn't exist
    output_directory = os.path.join(directory, 'model_info')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    models_directory = os.path.join(directory, 'models')

    for filename in os.listdir(models_directory):
        if filename.endswith(".joblib"):
            input_path = os.path.join(models_directory, filename)
            # Extracting the model's name from the filename and appending "_joblib" for the output filename
            output_filename = os.path.splitext(filename)[0] + "_joblib.txt"
            output_path = os.path.join(output_directory, output_filename)
            subprocess.run(['python', 'output_joblib.py', input_path, output_path])

if __name__ == '__main__':
    directory = "../results/pi2b/tf_pose_estimation/regression/kmeans_robust_3"
    run_for_all_models(directory)
