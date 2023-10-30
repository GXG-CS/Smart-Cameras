import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline

# Function to load data from a given file path
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Function to get the appropriate scaler based on the preprocessing name
def get_scaler(preprocess_name):
    if preprocess_name == "robust":
        return RobustScaler()
    elif preprocess_name == "standard":
        return StandardScaler()
    elif preprocess_name == "minmax":
        return MinMaxScaler()
    elif preprocess_name == "boxcox":
        return PowerTransformer(method='box-cox')
    elif preprocess_name == "yeojohnson":
        return PowerTransformer(method='yeo-johnson')
    else:
        return None

# Function to apply preprocessing and PCA
def preprocess_and_pca(data, preprocess_name):
    scaler = get_scaler(preprocess_name)
    if scaler is None:
        return None, None
    
    pipeline = Pipeline([
        ('scaler', scaler),
        ('pca', PCA(n_components=2))
    ])
    
    X = data.drop('label', axis=1)
    y = data['label']
    X_pca = pipeline.fit_transform(X)

    # Added this line to print the explained variance ratios
    pca = pipeline.named_steps['pca']  # Access the 'pca' step of the pipeline
    print("Explained variance ratios:", pca.explained_variance_ratio_)  # Print explained variance
    
    return X_pca[:, 0], X_pca[:, 1]

# Main code starts here
if __name__ == "__main__":
    root_dir = "../results/pi0/tf_pose_estimation/cluster/"
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
    fig, axs = plt.subplots(rows, cols, figsize=(60, 50))
    global_index = 0
    
    for method in methods:
        method_dir = os.path.join(root_dir, method)
        for file in os.listdir(method_dir):
            file_path = os.path.join(method_dir, file)
            data = load_data(file_path)
            if data is None:
                continue
            
            preprocess_name = file.split("_")[0]
            X_pca, Y_pca = preprocess_and_pca(data, preprocess_name)
            if X_pca is None or Y_pca is None:
                continue
            
            ax = axs[global_index // cols, global_index % cols]
            sc = ax.scatter(X_pca, Y_pca, s=3, c=data['label'], cmap='rainbow')
            ax.set_title(f"{abbreviations[method]}_{preprocess_name}_{file.split('_')[-1].split('.')[0]}")
            global_index += 1

     # Save the plot to a file before displaying it
    save_path = "../visualization/results_2D/clustering_visualization_p0.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)

    plt.show()
