# Load the necessary packages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#----------DATA UNDERSTANDING

# Load the CSV dataset using an GitHub URL
url = 'https://raw.githubusercontent.com/orangecapybara/orange/main/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
data = pd.read_csv(url)

##----------DATA PROCESSING

# Separate features and target
X = data.drop('Diabetes_binary', axis=1)
y = data['Diabetes_binary']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #test size 0.2 means 20% of the dataset is for testing and 80% is for training, and random state is to ensure random number generation

# Standardize the features (important for Gaussian Naive Bayes if features vary in scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

##----------MODELING AND EVALUATION

# Instantiate and train the Gaussian Naive Bayes model
bayes = GaussianNB()
bayes.fit(X_train_scaled, y_train)

# Apply the model into the test features using the predict() function to generate an array of predictions
y_pred = bayes.predict(X_test_scaled)

# Evaluate the performance of the model using the confusion matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
conf_matrix = confusion_matrix(y_test, y_pred)
labelNames = pd.Series(['diabetes', 'no diabetes'])
pd.DataFrame(conf_matrix,
     columns='Predicted ' + labelNames,
     index='Is ' + labelNames)
