import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# Load Models
model1_path = "../results/pi2b/tf_pose_estimation/regression/kmeans_robust_3/models/gbr_robust.joblib"
model2_path = "../results/pi3b/tf_pose_estimation/regression/kmeans_robust_3/models/gbr_robust.joblib"

model1 = joblib.load(model1_path)
model2 = joblib.load(model2_path)

# Feature Importances
importances1 = model1.feature_importances_
importances2 = model2.feature_importances_

# Suppose you have a list of feature names like this:
feature_names = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

# Display and Save Combined Feature Importances
def plot_and_save_combined_feature_importances(importances1, importances2, title1, title2, filename):
    indices1 = np.argsort(importances1)[::-1]
    indices2 = np.argsort(importances2)[::-1]
    
    plt.figure(figsize=(14, 7))
    
    plt.bar(np.arange(len(importances1)) - 0.2, importances1[indices1], 0.4, label=f"{title1}", align='center')
    plt.bar(np.arange(len(importances2)) + 0.2, importances2[indices2], 0.4, label=f"{title2}", align='center')

    plt.title("Feature Importances Comparison")
    plt.xlabel("Feature Names")
    plt.ylabel("Importance")
    plt.xticks(range(len(importances1)), [feature_names[i] for i in indices1])
    plt.xlim([-1, len(importances1)])
    plt.legend(loc='upper left')
    
    # Move clustering and regression model info to the right side
    plt.text(len(importances1) - 0.5, max(importances1) * 0.5, "Clustering Model: kmeans_robust_3\nRegression Model: gbr_robust", 
             bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Set output directory and filename for the combined feature importances image
output_dir = "../results/compare2models/3b_2b"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.join(output_dir, "Combined_Feature_Importances.png")

plot_and_save_combined_feature_importances(
    importances1, importances2, 
    "Model 1 from pi2b", "Model 2 from pi3b", 
    output_filename
)
