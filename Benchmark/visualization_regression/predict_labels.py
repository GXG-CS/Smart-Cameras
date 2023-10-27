import argparse
import pandas as pd
import os
from joblib import load
from sklearn.preprocessing import PowerTransformer

def get_preprocessor(method, fitted_preprocessors=None):
    """
    Return the preprocessing transformation based on the specified method.
    """
    if method == 'yeojohnson':
        if fitted_preprocessors and 'yeojohnson' in fitted_preprocessors:
            return fitted_preprocessors['yeojohnson']
        else:
            return PowerTransformer(method='yeojohnson', standardize=True)
    # You can add more preprocessing methods here as needed
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

def predict_labels(model_path, test_data_path, save_path):
    # Load the object from the specified path
    data = load(model_path)

    # Check if the loaded data is a dictionary or just a model
    if isinstance(data, dict):
        model = data['model']
        fitted_preprocessors = data.get('preprocessors', {})
    else:  # assume it's the model directly
        model = data
        fitted_preprocessors = {}

    # Load the CSV data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('label', axis=1)

    # Extract preprocessing method from filename
    filename = os.path.basename(test_data_path)
    if 'yeojohnson' in filename:
        preprocessor = get_preprocessor('yeojohnson', fitted_preprocessors=fitted_preprocessors)
        X_test = preprocessor.transform(X_test)
    # Add more conditions here for other preprocessing methods if needed

    # Predict using the model
    y_pred = model.predict(X_test)

    # Add the predictions to the dataframe
    test_data['predicted_label'] = y_pred

    # Save the dataframe with predictions to a CSV file
    test_data.to_csv(save_path, index=False)


def main():
    parser = argparse.ArgumentParser(description='Predict Labels for Test Data and Save Predictions.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the .joblib model file.')
    parser.add_argument('--test_data', required=True, type=str, help='Path to the test data file (.csv format).')
    parser.add_argument('--save_path', required=True, type=str, help='Path to save the prediction results.')
    
    args = parser.parse_args()
    
    predict_labels(args.model_path, args.test_data, args.save_path)

if __name__ == "__main__":
    main()



# python3 ./visualization_regression/predict_labels.py --model_path ./results/pi3b/tf_pose_estimation/regression/gaussian_mixture_yeojohnson_4/models/gbr_robust.joblib --test_data ./data_cleaning/pi3b/pi3b_yeojohnson_4_filtered.csv --save_path ./visualization_regression/results/prediction_results.csv
