from joblib import load
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Prediction script.")
    parser.add_argument('--data', type=str, required=True, help="Path to the data file for prediction.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model file.")
    parser.add_argument('--output', type=str, required=True, help="Output directory path for storing the predictions.")
    
    args = parser.parse_args()

    # Check if the output directory exists, if not, create it
    output_dir = args.output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load the model
    model = load(args.model)
    
    # Load the data
    data = pd.read_csv(args.data)
    X = data.drop('label', axis=1)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions, columns=['prediction'])
    predictions_file_path = os.path.join(output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_file_path, index=False)
    
    print(f"Predictions saved to {predictions_file_path}")

if __name__ == "__main__":
    main()
