import argparse
import pandas as pd
import os
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)

def ensure_directory_exists(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def check_single_cluster(labels):
    """
    Check if all data points are assigned to a single cluster.
    :param labels: List or array of cluster labels.
    :return: True if all data points are in a single cluster, False otherwise.
    """
    unique_labels = set(labels)
    return len(unique_labels) == 1

def evaluate_clustering(data_path, output_path):
    # Ensure output directory exists
    ensure_directory_exists(output_path)

    # Load data
    data = pd.read_csv(data_path)
    
    # Check if 'label' column exists in the data
    if 'label' not in data.columns:
        raise ValueError("The CSV file should have a 'label' column indicating cluster assignments.")

    # Extract labels and drop the 'label' column to get feature data
    labels = data['label'].values
    data.drop('label', axis=1, inplace=True)
    
    # Check for single cluster assignment
    if check_single_cluster(labels):
        print(f"All data points in {data_path} are assigned to a single cluster.")
        return
    
    try:
        # Calculate metrics
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
    
        # Determine the name of the output file based on the input data filename
        base_name = os.path.basename(data_path).split('.')[0]
        evaluation_file_name = f"{base_name}_evaluation.txt"
        full_output_path = os.path.join(output_path, evaluation_file_name)
        
        # Save results to output file
        with open(full_output_path, 'w') as f:
            f.write(f"Silhouette Score: {silhouette}\n")
            f.write(f"Davies-Bouldin Index: {davies_bouldin}\n")
            f.write(f"Calinski-Harabasz Score: {calinski_harabasz}\n")
    
    except ValueError as e:
        if str(e) == "Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)":
            print(f"Error with file {data_path}: All data points assigned to a single cluster.")
        else:
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering results.")
    parser.add_argument('--data', type=str, required=True, help="Path to the cluster result CSV file.")
    parser.add_argument('--output', type=str, required=True, help="Directory to save the evaluation results.")
    
    args = parser.parse_args()
    evaluate_clustering(args.data, args.output)
