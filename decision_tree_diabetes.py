import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
data = pd.read_csv('diabetes.csv')

# # Check for missing values
# print(data.isnull().sum())

# Split the dataset into features and target variable
X = data[['Pregnancies',	'Glucose',	'BloodPressure',	'SkinThickness',	'Insulin',	'BMI'	,'DiabetesPedigreeFunction'	,'Age']]  # Features
y = data['Outcome']                 # Target variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a decision tree classifier
# Create a decision tree classifier with pre-pruning
model = DecisionTreeClassifier(max_depth=3, min_samples_split=5, min_samples_leaf=3)
# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(12,8))
tree.plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'])
plt.show()