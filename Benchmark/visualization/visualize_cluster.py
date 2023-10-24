import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, MinMaxScaler

# Directories and methods
root_dir = "../results/pi2b/tf_pose_estimation/cluster/"
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

# Figure settings
rows, cols = 8, 8  # Adjusted based on the number of sub-figures
fig, axs = plt.subplots(rows, cols, figsize=(60, 50))
global_index = 0

for method in methods:
    method_dir = os.path.join(root_dir, method)
    for file in os.listdir(method_dir):
        file_path = os.path.join(method_dir, file)
        data = pd.read_csv(file_path)
        X = data.iloc[:, :-1].values
        
        # Preprocessing based on filename
        if "robust" in file:
            scaler = RobustScaler()
            preprocess_name = "robust"
        elif "standard" in file:
            scaler = StandardScaler()
            preprocess_name = "standard"
        elif "boxcox" in file:
            X = X + 1e-10  # Ensuring all values are positive
            scaler = PowerTransformer(method='box-cox')
            preprocess_name = "boxcox"
        elif "minmax" in file:
            scaler = MinMaxScaler()
            preprocess_name = "minmax"
        elif "yeojohnson" in file:
            scaler = PowerTransformer(method='yeo-johnson')
            preprocess_name = "yeojohnson"
        else:
            scaler = StandardScaler()
            preprocess_name = "standard"

        X_scaled = scaler.fit_transform(X)
        
        # Applying PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        ax = axs[global_index // cols, global_index % cols]
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=3, c=data['label'], cmap='viridis')
        ax.set_title(f"{abbreviations[method]}_{preprocess_name}_{file.split('_')[-1].split('.')[0]}")
        global_index += 1

# Save the figure
save_path = "../visualization/results/clustering_visualization_p2.png"
plt.tight_layout()
plt.savefig(save_path)
plt.show()
