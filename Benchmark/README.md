Benchmark contains three modules:
1. Data_collection module
2. Clustering module
3. Regression module

Folder structure:

Benchmark/
├── data_collection/
│   ├── pi0/
│   ├── pi2b/
│   ├── pi3b/
│   ├── pi4b/
│   └── tensor_flow/
│
├── results/
│   └── pi3b/
│       └── tf_pose_estimation/
│           ├── cluster/
│           │   ├── affinity_propagation/
│           │   ├── agglomerative/
│           │   ├── birch/
│           │   ├── dbscan/
│           │   ├── gaussian_mixture/
│           │   ├── kmeans/
│           │   ├── mean_shift/
│           │   ├── minibatch_kmeans/
│           │   ├── optics/
│           │   ├── spectral/
│           │   
│           └── [other results]
│
├── script/
│   ├── eval_cluster_summary.py
│   ├── eval_cluster_txt.py
│   └── run_cluster.py
│
├── visualization/
│   └── visualize_cluster.py
│
├── cluster.py
└── regression.py




Data Collection: Gather data from experiments adjusting sdram_freq, cpu_req, cpu_cores, and mem_limit_kb to measure avg_fps and total_time.

Data Preprocessing: Clean the data, handle missing values, and normalize or standardize if necessary.

Clustering: Use clustering algorithms (e.g., KMeans, DBSCAN) to segment the dataset into different performance levels based on avg_fps and total_time. Assign labels to each data point based on the cluster it belongs to.

Classification (Optional):

Split the labeled data into training and test sets.
Train classification models (e.g., Logistic Regression, Decision Trees) on the training set.
Validate the model's accuracy on the test set to confirm the distinctiveness of the clusters.
Use the trained model to predict performance levels for new configurations.
Regression: For each cluster identified in the clustering step:

Use the configuration parameters (sdram_freq, cpu_req, etc.) as features.
Use avg_fps and total_time as targets.
Train regression models (e.g., Linear Regression, Random Forest Regression) to predict avg_fps and total_time based on configuration parameters.
Evaluation: Measure the performance of regression models using metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).

Deployment: Implement the best-performing models in the desired application to predict performance outcomes for various configurations.

Iterate: Periodically revisit the models, especially if new data becomes available or if the system's behavior changes over time. Adjust and retrain models as necessary.



data_collection steps: (pi3b_tf_pose_estimation)
1. sudo nano /boot/config.txt     sdram_freq=450
2. crontab -e @reboot xxx
3. change counter.txt to 0
4. change sdram_freq.txt sdram_freq list
5. debug info: log.txt and error_log.txt
6. results.txt stores the results of [cpu_cores, cpu_freq, avg_fps, sdram_freq, total_time, mem_limit_kb]
7. ./4_metrics.sh







