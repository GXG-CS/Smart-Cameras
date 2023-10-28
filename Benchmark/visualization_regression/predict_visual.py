import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import numpy as np

def visualize_predictions(csv_file, output_dir, jitter_strength=0.02):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure required columns are present
    if 'Predicted' not in df.columns or 'True' not in df.columns:
        print("Error: CSV does not contain expected columns.")
        return

    # Extract Predicted and True values and add jitter
    predicted_values = df['Predicted'].values + np.random.randn(len(df['Predicted'].values)) * jitter_strength
    true_values = df['True'].values + np.random.randn(len(df['True'].values)) * jitter_strength
    errors = predicted_values - true_values

    # Plot
    plt.figure(figsize=(10, 10))
    plt.scatter(true_values, predicted_values, s=100, c='blue', alpha=0.5, label='Data Points')  # Increase the s value for larger points
    plt.plot([0, 2], [0, 2], color='red', linestyle='-', linewidth=1, label='Identity Line')
    
    # Highlight points with large errors
    over_predictions = errors > 0.1  # Adjust threshold if necessary
    under_predictions = errors < -0.1  # Adjust threshold if necessary
    plt.scatter(true_values[over_predictions], predicted_values[over_predictions], color='red', label='Over-Predictions')
    plt.scatter(true_values[under_predictions], predicted_values[under_predictions], color='purple', label='Under-Predictions')

    # Labelling the plot
    plt.title('Predicted vs True Values with Jitter')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(output_dir, os.path.basename(csv_file).replace('.csv', '_jitter_plot.png'))
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize regression predictions with jitter.')
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV file containing predictions.')
    parser.add_argument('--output_dir', type=str, default='./', help='Directory to save the output plot.')
    parser.add_argument('--jitter', type=float, default=0.02, help='Strength of jitter to add to data points. Default is 0.02.')

    args = parser.parse_args()
    visualize_predictions(args.data, args.output_dir, jitter_strength=args.jitter)



# python predict_visual.py ../results/pi3b/tf_pose_estimation/regression/gaussian_mixture_yeojohnson_4/prediction_results/gbr_robust.joblib_results.csv ./results
