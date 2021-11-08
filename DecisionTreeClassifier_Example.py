import pandas as pd
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv('car_evaluation.csv')
features = data.columns[:-1]

# Split training and test data
train = data.iloc[:-350]
test = data.iloc[-350:]
test.reset_index(drop=True, inplace=True)

# Split data into features and labels
X_train = train[features]
y_train = train['class']

X_test = test[features]
y_test = test['class']

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
print(confusion_matrix(y_test, y_pred, labels=list(set(y_test))))
