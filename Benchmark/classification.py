import os
import argparse
import pandas as pd
from preprocessing import preprocess_data
from evaluation import calculate_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Importing classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def get_classifier(method):
    classifiers = {
        'logistic_regression': LogisticRegression(),
        'knn': KNeighborsClassifier(),
        'svm': SVC(),
        'decision_tree': DecisionTreeClassifier(),
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GradientBoostingClassifier(),
        'naive_bayes': GaussianNB(),
        'mlp': MLPClassifier(),
        'adaboost': AdaBoostClassifier(),
        'qda': QuadraticDiscriminantAnalysis()
    }
    return classifiers.get(method, None)

def main():
    parser = argparse.ArgumentParser(description="Benchmark classification script.")
    parser.add_argument('--data', type=str, default=os.path.join('data', 'pi3b_tf_pose_estimation_results.csv'), help="Relative path to the data file.")
    parser.add_argument('--method', type=str, required=True, choices=['logistic_regression', 'knn', 'svm', 'decision_tree', 'random_forest', 'gradient_boosting', 'naive_bayes', 'mlp', 'adaboost', 'qda'], help="Classification method to use.")
    parser.add_argument('--preprocess', type=str, required=True, help="Preprocessing method to use from preprocessing.py.")
    parser.add_argument('--metric', type=str, required=True, help="Metric to use from evaluation.py.")
    
    args = parser.parse_args()

    # Load and preprocess the data
    data = pd.read_csv(args.data)
    X, y = preprocess_data(data, method=args.preprocess, labels=['avg_fps', 'total_time'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get the classifier
    classifier = get_classifier(args.method)
    if not classifier:
        print(f"Invalid method: {args.method}")
        return

    # Train and predict
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    other_metrics = calculate_metrics(y_test, y_pred, method=args.metric)
    
    print(f"Accuracy: {accuracy}")
    print(other_metrics)

if __name__ == "__main__":
    main()
