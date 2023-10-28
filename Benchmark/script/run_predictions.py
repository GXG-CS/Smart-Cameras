import os
import subprocess

def run_prediction_for_model(model_path, data_path, output_path):
    """
    Function to run the prediction script for a specific model.
    
    Parameters:
    - model_path: Path to the saved model (pipeline).
    - data_path: Path to the input data for prediction.
    - output_path: Path to save the prediction results.
    """
    command = [
        'python', 'prediction.py',
        '--model', model_path,
        '--data', data_path,
        '--output', output_path
    ]
    
    subprocess.run(command)

def run_all_models(model_dir, data_file, results_dir):
    """
    Run predictions on all models in the model directory using the data from the data file.
    """
    # Ensure the results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        output_filename = os.path.join(results_dir, f"{model_name}_results.csv")
        
        run_prediction_for_model(model_path, data_file, output_filename)

if __name__ == '__main__':
    # Relative Paths
    model_dir = "../results/pi4b/tf_pose_estimation/regression/optics_standard_9/models"
    data_file = "../data_cleaning/pi4b/pi4b_standard_9_filtered.csv"
    results_dir = "../results/pi4b/tf_pose_estimation/regression/optics_standard_9/prediction_results"

    run_all_models(model_dir, data_file, results_dir)

    print(f"Results saved in {results_dir}")
