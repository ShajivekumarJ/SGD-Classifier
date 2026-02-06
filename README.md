# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and split it into features and target labels.
2. Split the dataset into training and testing sets and normalize the features.
3. Train the SGD Classifier using logistic regression loss.
4. Predict the species and evaluate the model using confusion matrix, accuracy, precision, recall, and classification report.

## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SHAJIVE KUMAR J
RegisterNumber:  212225230158
```
~~~
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SGD Classifier with Logistic Regression loss
model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

~~~

## Output:
![alt text](<ML_EX-7 OUTPUT.png>)


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
