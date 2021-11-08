import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('car_evaluation.csv')
features = data.columns[:-1]

# Split training and test data
train = data.iloc[:-350]
test = data.iloc[-350:]
test.reset_index(drop=True, inplace=True)

# Split data into features and labels
#X_train = train[features]
#y_train = train['class']

#X_test = test[features]
#y_test = test['class']

X_train, X_test, y_train, y_test = train_test_split(data[features], data['class'], 
                                                    test_size=0.05, random_state=42)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

# Initialize the Classifier
clf = DecisionTreeClassifier()
# Fit to training data
clf.fit(X_train, y_train)
# Evaluate accuracy
training_accuracy = clf.evaluate(X_train, y_train)
test_accuracy = clf.evaluate(X_test, y_test)
print("Training Accuracy: ", training_accuracy)
print("Test Accuracy: ", test_accuracy)

y_pred = y_test.copy()
for index, _ in X_test.iterrows(): 
    y_pred.iloc[index] = clf.predict(X_test.iloc[index])
    result = clf.predict(X_test.iloc[index])
confusion_matrix(y_test.tolist(), y_pred.tolist())