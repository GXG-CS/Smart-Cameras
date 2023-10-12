from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

def run_svm(df):
    """
    Runs Support Vector Machines (SVM) classification on the provided DataFrame.
    
    Parameters:
    df (DataFrame): the dataset
    
    Returns:
    float: the accuracy of the classifier
    """
  
    # Assuming the last column is the target variable and the others are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
  
    # Splitting the data into 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
  
    # Create a Support Vector Machine classifier object
    svm_classifier = SVC(kernel='linear')
  
    # Fitting the model
    svm_classifier.fit(X_train, y_train)
  
    # Making predictions
    y_pred = svm_classifier.predict(X_test)
  
    # Evaluating the classifier
    accuracy = accuracy_score(y_test, y_pred)
  
    print(f"SVM Classifier Accuracy: {accuracy}")
  
    return accuracy
