import argparse
import os
from datetime import datetime
import pandas as pd

def get_clusterer(method, n_clusters=3):
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters)
    elif method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'affinity_propagation':
        from sklearn.cluster import AffinityPropagation
        return AffinityPropagation()
    elif method == 'spectral':
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(n_clusters=n_clusters)
    elif method == 'dbscan':
        from sklearn.cluster import DBSCAN
        return DBSCAN()
    elif method == 'optics':
        from sklearn.cluster import OPTICS
        return OPTICS()
    elif method == 'birch':
        from sklearn.cluster import Birch
        return Birch(n_clusters=n_clusters)
    elif method == 'mean_shift':
        from sklearn.cluster import MeanShift
        return MeanShift()
    elif method == 'minibatch_kmeans':
        from sklearn.cluster import MiniBatchKMeans
        return MiniBatchKMeans(n_clusters=n_clusters)
    elif method == 'gaussian_mixture':
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=n_clusters)
    else:
        print(f"Invalid method: {method}")
        return None

def get_preprocessor(method):
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    elif method == 'yeojohnson':
        from sklearn.preprocessing import PowerTransformer
        return PowerTransformer(method='yeo-johnson')
    elif method == 'boxcox':
        from sklearn.preprocessing import PowerTransformer
        return PowerTransformer(method='box-cox')
    elif method == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(output_distribution='normal')
    else:
        print(f"Invalid preprocessing method: {method}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark clustering script.")
    parser.add_argument('--data', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--method', type=str, required=True, choices=['kmeans', 'agglomerative'], help="Clustering method to use.")  # Shortened for brevity
    parser.add_argument('--n_clusters', type=int, default=3, help="Number of clusters. Used for algorithms that need the specification of cluster number.")
    parser.add_argument('--preprocessing', type=str, choices=['standard', 'minmax'], default='standard', help="Preprocessing method to use.")  # Shortened for brevity
    parser.add_argument('--output', type=str, help="Output directory path for clustered results.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the data
    data = pd.read_csv(args.data)

    # Apply preprocessing
    preprocessor = get_preprocessor(args.preprocessing)
    processed_data = preprocessor.fit_transform(data)

    # Get the clusterer
    clusterer = get_clusterer(args.method, args.n_clusters)

    # Apply clustering
    labels = clusterer.fit_predict(processed_data)
    data['label'] = labels

    # Save the clustered data
    input_name = os.path.basename(args.data).split('.')[0]
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"{input_name}_{current_time}.csv"
    output_path = os.path.join(args.output, output_filename)
    data.to_csv(output_path, index=False)
    
    print(f"Clustered results saved to {output_path}")

if __name__ == "__main__":
    main()
