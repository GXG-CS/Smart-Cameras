import joblib
import argparse
import os

def output_model_info(model_path, output_file):
    """
    Outputs all the attributes and their values for a given model loaded with joblib.
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Extracting model attributes
    attributes = dir(model)
    
    # Filter out magic methods and return only attributes
    attributes = [attr for attr in attributes if not attr.startswith("__")]
    
    with open(output_file, 'w') as f:
        for attr in attributes:
            try:
                value = getattr(model, attr)
                
                # If the value is too long, we truncate it to make it readable
                if isinstance(value, (list, dict, str)) and len(value) > 100:
                    value = str(value)[:100] + "..."
                
                f.write(f"{attr}: {value}\n")
            except Exception as e:
                f.write(f"{attr}: Error retrieving attribute - {e}\n")
        f.write("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Output detailed information about a model saved with joblib.")
    parser.add_argument("input_path", help="Path to the input model file.")
    parser.add_argument("output_path", help="Path to the output file where detailed model information will be saved.")
    
    args = parser.parse_args()
    
    # Check if the provided input path exists
    if not os.path.exists(args.input_path):
        print(f"The provided model path '{args.input_path}' does not exist.")
        exit(1)

    output_model_info(args.input_path, args.output_path)
    print(f"Model information saved to {args.output_path}.")

# python output_joblib.py ../results/pi3b/tf_pose_estimation/regression/kmeans_robust_3/models/gbr_robust.joblib ../results/pi3b/tf_pose_estimation/regression/kmeans_robust_3/models/gbr_robust.txt