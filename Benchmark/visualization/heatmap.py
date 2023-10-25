import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

def main():
    root_dir = "../results/pi3b/tf_pose_estimation/cluster/"
    results_dir = "/Users/xiaoguangguo/Documents/Smart-Cameras/Benchmark/visualization/results_heatmap"
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

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

    for method in methods:
        method_dir = os.path.join(root_dir, method)
        for file in os.listdir(method_dir):
            file_path = os.path.join(method_dir, file)
            data = load_data(file_path)
            
            preprocess_name = file.split("_")[0]
            X_scaled, Y_scaled = preprocess_data(data, preprocess_name)
            
            # Combine scaled features and labels into a new DataFrame
            df = pd.DataFrame(np.hstack((X_scaled, Y_scaled)), 
                              columns=['sdram_freq', 'cpu_cores', 'cpu_freq', 'mem_limit_kb', 'avg_fps', 'total_time'])
            df['label'] = data['label']

            # Create a pair plot colored by cluster labels
            sns.pairplot(df, hue='label', diag_kind="hist")
            plt.title(f"{abbreviations[method]}_{preprocess_name}_{file.split('_')[-1].split('.')[0]}")
            
            # Save the plot
            save_path = os.path.join(results_dir, f"{abbreviations[method]}_{preprocess_name}_{file.split('_')[-1].split('.')[0]}_pairplot.png")
            plt.savefig(save_path)
            plt.close()

            # Create a heatmap for correlations
            corr_matrix = df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f"{abbreviations[method]}_{preprocess_name}_{file.split('_')[-1].split('.')[0]}_Heatmap")
            
            # Save the heatmap
            heatmap_path = os.path.join(results_dir, f"{abbreviations[method]}_{preprocess_name}_{file.split('_')[-1].split('.')[0]}_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close()

if __name__ == "__main__":
    main()
