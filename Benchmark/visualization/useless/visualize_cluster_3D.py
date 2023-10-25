import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, MinMaxScaler

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_data(data, preprocess_name):
    X = data[['sdram_freq', 'cpu_cores', 'cpu_freq', 'mem_limit_kb']].values
    Y = data[['avg_fps', 'total_time']].values

    scaler = get_scaler(preprocess_name)
    X_scaled = scaler.fit_transform(X)
    Y_scaled = scaler.fit_transform(Y)

    return X_scaled, Y_scaled

def get_scaler(preprocess_name):
    if preprocess_name == "robust":
        return RobustScaler()
    elif preprocess_name == "standard":
        return StandardScaler()
    elif preprocess_name == "boxcox":
        return PowerTransformer(method='box-cox')
    elif preprocess_name == "minmax":
        return MinMaxScaler()
    elif preprocess_name == "yeojohnson":
        return PowerTransformer(method='yeo-johnson')
    else:
        return StandardScaler()

def apply_pca(X, Y):
    n_components_X = min(3, X.shape[1], X.shape[0]-1)
    n_components_Y = min(3, Y.shape[1], Y.shape[0]-1)
    
    pca_X = PCA(n_components=n_components_X)
    pca_Y = PCA(n_components=n_components_Y)
    
    return pca_X.fit_transform(X), pca_Y.fit_transform(Y)

def main():
    root_dir = "../results/pi3b/tf_pose_estimation/cluster/"
    
    methods = [
        "affinity_propagation", "agglomerative", "birch", "dbscan", 
        "gaussian_mixture", "kmeans", "mean_shift", "minibatch_kmeans", 
        "optics", "spectral"
    ]

    abbreviations = {
        "affinity_propagation": "AP",
        "agglomerative": "AG",
        "birch": "BI",
        "dbscan": "DB",
        "gaussian_mixture": "GM",
        "kmeans": "KM",
        "mean_shift": "MS",
        "minibatch_kmeans": "MK",
        "optics": "OP",
        "spectral": "SP"
    }

    files_count = sum([len(os.listdir(os.path.join(root_dir, method))) for method in methods])
    rows, cols = (files_count // 13) + 1, 13
    fig = plt.figure(figsize=(60, 50))
    
    global_index = 0

    for method in methods:
        method_dir = os.path.join(root_dir, method)
        for file in os.listdir(method_dir):
            file_path = os.path.join(method_dir, file)
            data = load_data(file_path)
            
            preprocess_name = file.split("_")[0]
            X_scaled, Y_scaled = preprocess_data(data, preprocess_name)
            X_pca, Y_pca = apply_pca(X_scaled, Y_scaled)
            
            ax = fig.add_subplot(rows, cols, global_index+1, projection='3d')
            sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], s=3, c=data['label'], cmap='rainbow')
            ax.set_title(f"{abbreviations[method]}_{preprocess_name}_{file.split('_')[-1].split('.')[0]}")
            global_index += 1

    save_path = "../visualization/results_3D/clustering_visualization_p3.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main()
