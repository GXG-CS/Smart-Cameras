import argparse
import pandas as pd
import os
from datetime import datetime
from sklearn.cluster import (KMeans, AgglomerativeClustering, AffinityPropagation, SpectralClustering,
                             DBSCAN, OPTICS, Birch, MeanShift, MiniBatchKMeans)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def get_clusterer(method, n_clusters=3):
    clusterers = {
        'kmeans': KMeans(n_clusters=n_clusters),
        'agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'affinity_propagation': AffinityPropagation(),
        'spectral': SpectralClustering(n_clusters=n_clusters),
        'dbscan': DBSCAN(),
        'optics': OPTICS(),
        'birch': Birch(n_clusters=n_clusters),
        'mean_shift': MeanShift(),
        'minibatch_kmeans': MiniBatchKMeans(n_clusters=n_clusters),
        'gaussian_mixture': GaussianMixture(n_components=n_clusters)
    }
    return clusterers.get(method, None)

def main():
    parser = argparse.ArgumentParser(description="Benchmark clustering script.")
    parser.add_argument('--data', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--method', type=str, required=True, choices=['kmeans', 'agglomerative', 'affinity_propagation', 'spectral', 'dbscan', 'optics', 'birch', 'mean_shift', 'minibatch_kmeans', 'gaussian_mixture'], help="Clustering method to use.")
    parser.add_argument('--n_clusters', type=int, default=3, help="Number of clusters. Used for algorithms that need the specification of cluster number.")
    parser.add_argument('--output', type=str, help="Output directory path for clustered results.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load and preprocess the data
    data = pd.read_csv(args.data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Get the clusterer
    clusterer = get_clusterer(args.method, args.n_clusters)
    if not clusterer:
        print(f"Invalid method: {args.method}")
        return

    # Apply clustering
    labels = clusterer.fit_predict(scaled_data)
    data['label'] = labels

    # Save the clustered data
    input_name = os.path.basename(args.data).split('.')[0]
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{input_name}_cluster_{current_time}.csv"
    output_path = os.path.join(args.output, output_filename)
    data.to_csv(output_path, index=False)
    
    print(f"Clustered results saved to {output_path}")

if __name__ == "__main__":
    main()
