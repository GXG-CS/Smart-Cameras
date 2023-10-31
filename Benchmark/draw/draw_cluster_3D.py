import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_data(data_path, output_path):
    # Check if the directory exists, if not create it
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read data into a DataFrame
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print("The file was not found at the specified path.")
        return

    # Select features for visualization
    features = ['sdram_freq', 'cpu_cores', 'cpu_freq']
    # features = ['cpu_freq', 'cpu_cores', 'avg_fps']

    selected_data = df[features]

    # Get unique labels for coloring
    labels = df['label'].unique()

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point and color it based on its label
    for label in labels:
        label_data = selected_data[df['label'] == label]
        ax.scatter(label_data['sdram_freq'], label_data['cpu_cores'], label_data['cpu_freq'], label=f'Label {label}')
        # ax.scatter(label_data['cpu_freq'], label_data['cpu_cores'], label_data['avg_fps'], label=f'Label {label}')

    # Add axis labels
    # ax.set_xlabel('CPU Frequency')
    ax.set_xlabel('Sdram Frequency')
    ax.set_ylabel('CPU Cores')
    ax.set_zlabel('CPU Frequency')

    # Add legend
    ax.legend()

    # Save plot to output path
    plt.savefig(output_path)

    # Show the plot
    # plt.show()

if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Visualize selected features with different colors for each label.')

    # Add arguments for data path and output path
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV data file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output plot.')

    # Parse arguments
    args = parser.parse_args()

    # Call function to plot data
    plot_data(args.data_path, args.output_path)


# python draw_cluster_3D.py --data_path ../results/pi3b/tf_pose_estimation/cluster/gaussian_mixture/boxcox_4.csv --output_path ./output_plot.png
