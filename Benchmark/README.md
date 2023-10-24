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



Setup pi
1. miniconda python3.9
2. git clone https://github.com/GXG-CS/Smart-Cameras.git
3. tensorflow pose_estimation setup
4. Install GTK libraries: sudo apt-get install libgtk2.0-dev
5. sudo apt-get install xvfb
6. sudo apt-get install cpufrequtils

data_collection steps: (pi3b_tf_pose_estimation)
1. sudo nano /boot/config.txt     sdram_freq=450
2. crontab -e @reboot xxx
3. change counter.txt to 0
4. change sdram_freq.txt sdram_freq list
5. debug info: log.txt and error_log.txt
6. results.txt stores the results of [cpu_cores, cpu_freq, avg_fps, sdram_freq, total_time, mem_limit_kb]
7. ./4_metrics.sh


pi Zero
1. single-core
2. CPU freq 1GHz
3. 


pi 2b hardware configuration:
1. Quad Core CPU
2. CPU clock speed of 900 MHz
3. sdram_freq 450MHz
192.168.1.142

pi3b+
1. Quad Core CPU
2. CPU freq 1.4GHz
3. sdram_freq 450MHz
192.168.1.154

pi 4b
1. Quad Core CPU
2. CPU freq 1.8Ghz
3. sdram_freq
192.168.246

