# -*- coding: utf-8 -*-
"""kNN-Diabetes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PJAjY79WdyacFlry2WWkiMY0-ZjgI0Hl
"""

import pandas as pd
rawDF = pd.read_csv('Diabetes Binary.csv')
rawDF.info()

rawDF.head()

rawDF.describe()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Separate the features and the target variable
X = rawDF.drop('Diabetes_binary', axis=1)
y = rawDF['Diabetes_binary']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Choose an initial value for K
k = int(np.sqrt(len(y_train))/2)  # Adjusted to avoid a too-large K value

# Initialize and train the KNN model
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Predict on the testing set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

k, accuracy