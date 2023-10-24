import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

# Load Models
model1_path = "../results/pi2b/tf_pose_estimation/regression/kmeans_robust_3/models/gbr_robust.joblib"
model2_path = "../results/pi3b/tf_pose_estimation/regression/kmeans_robust_3/models/gbr_robust.joblib"

model1 = joblib.load(model1_path)
model2 = joblib.load(model2_path)

# Print model parameters
print("Parameters for Model 1:")
print(model1.get_params())

print("\nParameters for Model 2:")
print(model2.get_params())

if hasattr(model1, 'estimator_weights_'):
    print("\nEstimator weights for Model 1:")
    print(model1.estimator_weights_)

if hasattr(model2, 'estimator_weights_'):
    print("\nEstimator weights for Model 2:")
    print(model2.estimator_weights_)

# Feature Importances
importances1 = model1.feature_importances_
importances2 = model2.feature_importances_

# Using the feature names directly for plotting
feature_names = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

def plot_and_save_combined_feature_importances(importances1, importances2, title1, title2, filename):
    plt.figure(figsize=(14, 7))
    
    bar_width = 0.35
    n_features = len(importances1)
    index = np.arange(n_features)

    # Plotting bars
    plt.bar(index, importances1, bar_width, label=f"{title1}", align='center')
    plt.bar(index + bar_width, importances2, bar_width, label=f"{title2}", align='center')

    # Labels and title
    plt.title("Feature Importances Comparison")
    plt.xlabel("Feature Names")
    plt.ylabel("Importance")
    plt.xticks(index + bar_width/2, feature_names)  # position the feature names in the middle of the grouped bars
    plt.legend(loc='upper right')
    
    # Display clustering and regression model info to the right side of the plot
    ax = plt.gca()
    text_position = (n_features, ax.get_ylim()[1]*0.95)
    plt.text(text_position[0], text_position[1], "Clustering Model: kmeans_robust_3\nRegression Model: gbr_robust",
             horizontalalignment='right', verticalalignment='top', transform=ax.transData,
             bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))

    # Save and display
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
