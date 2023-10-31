import os
import subprocess

# Base directory where the CSV files are located
# base_directory = "../results/pi0/tf_pose_estimation/cluster"
base_directory = "show/different_cluster_methods/"

# Relative path to the directory where you want to save all output plots
# output_directory = "pi0/cluster_3D/"
output_directory = "show/different_cluster_methods/"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Find all CSV files in the base directory and its subdirectories
csv_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(base_directory) for f in filenames if f.endswith('.csv')]

# Loop through each CSV file and run draw_cluster_3D.py
for csv_file in csv_files:
    # Extract the filename to use it for saving the plot
    filename = os.path.basename(csv_file)
    filename_without_extension = os.path.splitext(filename)[0]

    # Construct the output path for each plot using the clustering method and filename
    clustering_method = os.path.basename(os.path.dirname(csv_file))
    output_path = os.path.join(output_directory, f"{clustering_method}_{filename_without_extension}_plot.png")

    # Run the draw_cluster_3D.py script with the current CSV file and output path
    subprocess.run(["python", "draw_cluster_3D.py", "--data_path", csv_file, "--output_path", output_path])

print(f"All plots have been saved in {output_directory}")

