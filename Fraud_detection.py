# Importing Packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Defining Function to load the dataset
def load_dataset(file_path):
    """
    Load the dataset from the specified file path.

    Parameters:
    file_path (str): The path to the CSV file containing the dataset.

    Returns:
    pandas.DataFrame: The loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

# Defining Function to summarize the dataset
def summarize_dataset(data):
    """
    Display a summary of the dataset, including its structure, information, and statistics.

    Parameters:
    data (pandas.DataFrame): The dataset to be summarized.
    """
    # Displaying the first 5rows of the dataset
    print("First 5 rows of the dataset:")
    print(data.head(5))
    
    # Displaying dataset information
    print("\nDataset information:")
    print(data.info())
    
    # Displaying summary statistics
    print("\nSummary statistics:")
    print(data.describe())

# Loading the dataset
file_path = r'C:\Users\R.ARJUN\OneDrive\Desktop\arjunp\Machine Learning Internships\Fraud detection\fraud_detection_dataset.csv'
dataset = load_dataset(file_path)

# Summarizing the dataset
summarize_dataset(dataset)

# Define features and target
X = dataset.drop('Class', axis=1)  # Defining the X or Features 
y = dataset['Class']  # Defining the y or Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the decision tree classifier algorithm
DTC_Model = DecisionTreeClassifier() 
DTC_Model.fit(X_train, y_train)

y_pred = DTC_Model.predict(X_test)  # Make Prediction on testing data

# Evaluating  model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Printing evaluation of metrics
print("\n\nDecision Tree Metrics: \n")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
