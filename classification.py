"""
=========================================================
Classification Example
=========================================================
This example uses the `iris` dataset, in order to 
demonstrate a classification technique. 
Support Vector Classifier is used in the model.

The classification report and the confusion matrix are also generated
"""
print(__doc__)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import pandas as pd

# Get the data
iris = datasets.load_iris()

features = iris.data
labels = iris.target

# Split the data for training and testing
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.5)

# Train a Support Vector Classifier
sv_classifier = svm.SVC()
sv_classifier.fit(features_train, labels_train)

# Test the classifier
predictions = sv_classifier.predict(features_test)

# Evaluate
report = classification_report(labels_test, predictions,  target_names=iris.target_names)
print(report)

confusion_df = pd.DataFrame(confusion_matrix(labels_test, predictions),
    columns=["Predicted - " + str(class_name) for class_name in iris.target_names],
    index = ["Actual - " + str(class_name) for class_name in iris.target_names])

print(confusion_df)