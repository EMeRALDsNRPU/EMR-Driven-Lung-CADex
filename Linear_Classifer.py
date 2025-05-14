import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load dataset from CSV file
# Replace 'your_dataset.csv' with the path to your CSV file
data = pd.read_csv('cleaned_data_for_linear_classifer.csv')

# Assuming the last column is the target variable and the rest are features
X = data.iloc[:, :-1].values  # Select all columns except the last as features
y = data.iloc[:, -1].values  # Select the last column as the target

# Initialize the Logistic Regression classifier
classifier = LogisticRegression(solver='liblinear')  # Using 'liblinear' solver for binary classification

# Fit the model to your data
classifier.fit(X, y)

# Example: Predict the class of a new observation (replace with actual new observation)
# new_observation = [[value1, value2, ..., value8]]  # 8 feature values
# prediction = classifier.predict(new_observation)
# print(prediction)

# To predict new observations, replace the example with actual data.
