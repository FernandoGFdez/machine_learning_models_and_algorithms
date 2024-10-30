# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluate Models
log_accuracy = accuracy_score(y_test, y_pred_log)
svm_accuracy = accuracy_score(y_test, y_pred_svm)

print("Logistic Regression Accuracy:", log_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, y_pred_log))
print("\nClassification Report (SVM):\n", classification_report(y_test, y_pred_svm))
