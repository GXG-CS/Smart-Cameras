import argparse
import pandas as pd
from joblib import load

def predict_with_model(model_path, data_path, output_path):
    """
    Function to make predictions using the saved model.

    Parameters:
    - model_path: Path to the saved model (pipeline).
    - data_path: Path to the input data for prediction.
    - output_path: Path to save the prediction results.
    """
    # Load the saved pipeline (preprocessor and model)
    pipeline = load(model_path)
    
    # Read the input data
    data = pd.read_csv(data_path)
    X = data.drop('label', axis=1)
    y_true = data.get('label', None)
    
    # Make predictions
    y_pred = pipeline.predict(X)
    
    # Create a DataFrame for the predictions
    df_predictions = pd.DataFrame({"Predicted": y_pred})
    if y_true is not None:
        df_predictions["True"] = y_true

    # Save the predictions to the specified output path
    df_predictions.to_csv(output_path, index=False)

    # Inform the user
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction script using the trained models from regression.py.")
    parser.add_argument('--model', type=str, required=True, help="Path to the saved model (pipeline) file.")
    parser.add_argument('--data', type=str, required=True, help="Path to the input data for prediction.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the prediction results.")
    
    args = parser.parse_args()

    predict_with_model(args.model, args.data, args.output)

    # python prediction.py --model ../results/pi3b/tf_pose_estimation/regression/gaussian_mixture_yeojohnson_4/models/gbr_robust.joblib --data ../data_cleaning/pi3b/pi3b_yeojohnson_4_filtered.csv --output results/pi3b/tf_pose_estimation/regression/gaussian_mixture_yeojohnson_4/prediction/predicted_values.csv

