import os
import glob
import subprocess

def main():
    # Relative directory where the models are saved
    models_dir = "../results/pi3b/tf_pose_estimation/regression/kmeans_robust_3/models"
    
    # Relative directory where the predictions will be saved
    output_dir = "../results/predictions/3b_2b"
    
    # Relative path to the new dataset for prediction
    data_path = "../results/pi2b/tf_pose_estimation/cluster/kmeans/robust_3.csv"

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all .joblib files in the models directory
    model_files = glob.glob(f"{models_dir}/*.joblib")
    
    for model_file in model_files:
        model_name = os.path.basename(model_file).split('.')[0]

        # Command to run prediction.py
        cmd = f"python prediction.py --model {model_file} --data {data_path} --output {output_dir}/{model_name}_predictions.csv"

        # Log the command
        print(f"Running command: {cmd}")

        # Run the command
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to run {cmd}")

if __name__ == "__main__":
    main()
