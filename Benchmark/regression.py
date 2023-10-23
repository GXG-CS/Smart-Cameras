import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt

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
    
    args = parser.parse_args()

    # Ensure output directory exists
    if args.output and not os.path.exists(args.output):
        os.makedirs(args.output)

    # Load the data
    data = pd.read_csv(args.data)
    X = data.drop('label', axis=1)
    y = data['label']

    # Apply preprocessing
    if args.preprocessing:
        preprocessor = get_preprocessor(args.preprocessing)
        X = preprocessor.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the regressor
    regressor = get_regressor(args.method)

    # Apply regression
    regressor.fit(X_train, y_train)

    # Print learned parameters
    if args.method in ['linear', 'ridge', 'lasso']:
        print(f"Coefficients: {regressor.coef_}")
        print(f"Intercept: {regressor.intercept_}")
    elif args.method == 'random_forest' or args.method == 'gbr':
        print(f"Feature importances: {regressor.feature_importances_}")
    elif args.method == 'svr':
        print(f"Support vectors: {regressor.support_vectors_}")
        print(f"Dual Coefficients: {regressor.dual_coef_}")
    elif args.method == 'neural_network':
        for i, weights in enumerate(regressor.coefs_):
            print(f"Weights for layer {i}: {weights}")

    y_pred = regressor.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")

    if args.output:
        output_filename = f"{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = os.path.join(args.output, output_filename)
        pd.DataFrame({"True": y_test, "Predicted": y_pred}).to_csv(output_path, index=False)
        print(f"Regression results saved to {output_path}")

if __name__ == "__main__":
    main()
