import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("iris.data.csv")
size_mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['target'] = df['target'].map(size_mapping)
X=df[['sepal l',	'sepal w',	'petal l',	'petal w']]
y=df['target']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the base models
base_models = [
    ('decision_tree', DecisionTreeClassifier(random_state=0)),
    ('logistic_regression', LogisticRegression(max_iter=200, random_state=0))
]

# Define the meta-model
meta_model = LogisticRegression(random_state=0)

# Create the stacking classifier
stacking_classifier = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# Fit the stacking classifier on the training data
stacking_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = stacking_classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of Stacking Classifier: {accuracy:.2f}')