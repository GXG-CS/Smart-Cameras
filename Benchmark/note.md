folder structure:

Benchmark/
│
├── cluster.py - Entry point for all cluster tasks.
│
├── regression.py - Entry point for all regression tasks.
│
├── data/
│ └── pi3b_tf_pose_estimation_results.csv - Dataset used in tasks.
│
├── preprocessing.py - Data preprocessing functionalities.
│
└── evaluation.py - Evaluation metrics for the models.


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






shell script:

run cluster kmeans with n_cluster 6,7,8,9,10,11,12
also run eval with it
and gather all the eval results to one txt to compare



