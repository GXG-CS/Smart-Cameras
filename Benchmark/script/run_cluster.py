import os
import subprocess

# Define all the clustering methods and preprocessing techniques
CLUSTER_METHODS = [
    'kmeans', 'agglomerative', 'affinity_propagation', 'spectral',
    'dbscan', 'optics', 'birch', 'mean_shift', 'minibatch_kmeans', 'gaussian_mixture'
]

PREPROCESSING_TECHNIQUES = {
    'kmeans': ['standard', 'minmax', 'robust'],
    'dbscan': ['standard', 'robust'],
    'agglomerative': ['standard', 'robust'],
    'spectral': ['standard', 'minmax'],
    'optics': ['standard', 'robust'],
    'mean_shift': ['standard', 'minmax'],
    'affinity_propagation': ['standard'],
    'birch': ['standard', 'robust'],
    'gaussian_mixture': ['standard', 'yeojohnson', 'boxcox'],
    'minibatch_kmeans': ['standard']
}

N_CLUSTERS_RANGE = range(3, 10)

# Paths
DATA_PATH = "../data_collection/pi4b/pi4b_tf_pose_estimation_results.csv"
# CLUSTER_SCRIPT_PATH = "Benchmark/cluster.py"


def main():
    for method in CLUSTER_METHODS:
        # Update output directory based on method
        OUTPUT_DIR = os.path.join("..", "results", "pi4b", "tf_pose_estimation", "cluster", method)
        # Ensure the directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        for preprocessing in PREPROCESSING_TECHNIQUES[method]:
            for n_clusters in N_CLUSTERS_RANGE:
                cmd = [
                    'python', '../cluster.py',
                    '--data', DATA_PATH,
                    '--method', method,
                    '--preprocessing', preprocessing,
                    '--n_clusters', str(n_clusters),
                    '--output', os.path.join(OUTPUT_DIR)
                ]

                subprocess.run(cmd)

if __name__ == "__main__":
    main()
