from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Function to perform logistic regression
def logistic_regression(df):
    # Assume that the last column is 'label'
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features by removing the mean and scaling to unit variance
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Initialize the logistic regression model
    classifier = LogisticRegression(random_state=42)
    
    # Fit the model
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Evaluate the model
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

# Sample code to read data
# df = pd.read_csv('/path/to/your/data.csv')

# Uncomment below to run the logistic regression function
# logistic_regression(df)
