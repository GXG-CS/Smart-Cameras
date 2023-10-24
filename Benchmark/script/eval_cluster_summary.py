import os
import pandas as pd

# Directory where the clustering evaluation results are located (relative path)
DATA_DIR = '../results/pi2b/tf_pose_estimation/cluster_eval'
# Path for the summary CSV file (relative path)
OUTPUT_CSV_PATH = '../results/pi2b/tf_pose_estimation/cluster_eval/cluster_evaluation_summary.csv'

def extract_metrics_from_txt(file_path):
    """
    Extract clustering metrics from the evaluation .txt file.
    """
    metrics = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Silhouette Score' in line:
                metrics['Silhouette Score'] = float(line.split(":")[1].strip())
            elif 'Davies-Bouldin Index' in line:
                metrics['Davies-Bouldin Index'] = float(line.split(":")[1].strip())
            elif 'Calinski-Harabasz Score' in line:
                metrics['Calinski-Harabasz Score'] = float(line.split(":")[1].strip())
    return metrics

def main():
    summary = []
    
    # Iterate over subdirectories in DATA_DIR
    for method in os.listdir(DATA_DIR):
        method_dir = os.path.join(DATA_DIR, method)
        if os.path.isdir(method_dir):
            # Iterate over .txt files in the method directory
            for filename in os.listdir(method_dir):
                if filename.endswith('_evaluation.txt'):
                    full_path = os.path.join(method_dir, filename)
                    metrics = extract_metrics_from_txt(full_path)
                    
                    # Extract the cluster details from the filename and prefix it with the method
                    cluster_details = filename.replace('_evaluation.txt', '')
                    record = {
                        'cluster_model': f"{method}_{cluster_details}",
                        'Silhouette Score': metrics['Silhouette Score'],
                        'Davies-Bouldin Index': metrics['Davies-Bouldin Index'],
                        'Calinski-Harabasz Score': metrics['Calinski-Harabasz Score']
                    }
                    summary.append(record)
    
    # Create a DataFrame from the summary and save it to a CSV file
    df = pd.DataFrame(summary)
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Summary saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    main()
