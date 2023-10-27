import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

from joblib import dump

from sklearn.pipeline import Pipeline

def get_regressor(method):
    if method == 'linear':
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    elif method == 'ridge':
        from sklearn.linear_model import Ridge
        return Ridge()
    elif method == 'lasso':
        from sklearn.linear_model import Lasso
        return Lasso()
    elif method == 'decision_tree':
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor()
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor()
    elif method == 'svr':
        from sklearn.svm import SVR
        return SVR()
    elif method == 'knn':
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor()
    elif method == 'gbr':
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor()
    elif method == 'neural_network':
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor()
    else:
        print(f"Invalid regression method: {method}")
        return None

def get_preprocessor(method):
    if method == 'standard':
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        return MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    elif method == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        return MaxAbsScaler()
    elif method == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(output_distribution='uniform')
    elif method == 'yeojohnson':
        from sklearn.preprocessing import PowerTransformer
        return PowerTransformer(method='yeo-johnson')
    elif method == 'log':
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(np.log1p, validate=True)
    elif method == 'normalizer':
        from sklearn.preprocessing import Normalizer
        return Normalizer()
    elif method == 'binarizer':
        from sklearn.preprocessing import Binarizer
        return Binarizer(threshold=0.5)
    else:
        print(f"Invalid preprocessing method: {method}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark regression script.")
    parser.add_argument('--data', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--method', type=str, required=True, help="Regression method to use.")
    parser.add_argument('--preprocessing', type=str, help="Preprocessing method to use.")
    parser.add_argument('--output', type=str, help="Output directory path for regression results.")
    # parser.add_argument('--save_model_dir', type=str, default=None, help="Directory to save the model.")

    # Hyperparameters for each method
    parser.add_argument('--alpha', type=float, default=1.0, help="Alpha for Ridge and Lasso Regression.")
    parser.add_argument('--tree_depth', type=int, default=None, help="Max Depth for Decision Tree.")
    parser.add_argument('--rf_n_estimators', type=int, default=100, help="Number of estimators for Random Forest.")
    parser.add_argument('--svr_c', type=float, default=1.0, help="C parameter for SVR.")
    parser.add_argument('--knn_n_neighbors', type=int, default=5, help="Number of neighbors for KNN.")
    parser.add_argument('--gbr_n_estimators', type=int, default=100, help="Number of estimators for GBR.")
    parser.add_argument('--nn_hidden_layer_sizes', type=str, default='100', help="Hidden layer sizes for Neural Network.")

    args = parser.parse_args()

    output_dir = args.output
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_save_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    metrics_file_path = f"{args.method}_{args.preprocessing}.txt"
    if output_dir:
        metrics_file_path = os.path.join(output_dir, metrics_file_path)


    data = pd.read_csv(args.data)
    X = data.drop('label', axis=1)
    y = data['label']

    preprocessor = None
    if args.preprocessing:
        preprocessor = get_preprocessor(args.preprocessing)
        # X = preprocessor.fit_transform(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = get_regressor(args.method)

    # Construct the pipeline
    steps = []
    if preprocessor:
        steps.append(('preprocessor', preprocessor))
    steps.append(('regressor', regressor))
    pipeline = Pipeline(steps)

    # Apply hyperparameters based on the chosen method
    if args.method == 'ridge':
        regressor.set_params(alpha=args.alpha)
    elif args.method == 'lasso':
        regressor.set_params(alpha=args.alpha)
    elif args.method == 'decision_tree':
        regressor.set_params(max_depth=args.tree_depth)
    elif args.method == 'random_forest':
        regressor.set_params(n_estimators=args.rf_n_estimators)
    elif args.method == 'svr':
        regressor.set_params(C=args.svr_c)
    elif args.method == 'knn':
        regressor.set_params(n_neighbors=args.knn_n_neighbors)
    elif args.method == 'gbr':
        regressor.set_params(n_estimators=args.gbr_n_estimators)
    elif args.method == 'neural_network':
        hidden_layer_sizes = tuple(map(int, args.nn_hidden_layer_sizes.split(',')))
        regressor.set_params(hidden_layer_sizes=hidden_layer_sizes)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    # regressor.fit(X_train, y_train)

    # y_pred = regressor.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    with open(metrics_file_path, "a") as metrics_file:
        metrics_file.write(f"Method: {args.method}\n")
        metrics_file.write(f"Hyperparameters: {regressor.get_params()}\n")
        metrics_file.write(f"Mean Squared Error: {mse}\n")
        metrics_file.write(f"Root Mean Squared Error: {rmse}\n")
        metrics_file.write(f"Mean Absolute Error: {mae}\n")
        metrics_file.write("="*30 + "\n")

    print(f"Metrics and parameters saved to {metrics_file_path}")

    # model_save_path = os.path.join(model_save_dir, f"{args.method}_{args.preprocessing}.joblib")
    # dump(regressor, model_save_path)
    # print(f"Model saved to {model_save_path}")

    model_save_path = os.path.join(model_save_dir, f"{args.method}_{args.preprocessing}.joblib")
    dump(pipeline, model_save_path)
    print(f"Pipeline (including preprocessor and model) saved to {model_save_path}")


if __name__ == "__main__":
    main()


# python regression.py --data results/pi3b/tf_pose_estimation/cluster/kmeans/robust_3.csv --method svr --preprocessing standard --output results/pi3b/tf_pose_estimation/regression/kmeans_robust_3
