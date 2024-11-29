import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the dataset
df = pd.read_csv("iris.data.csv")

# Map target labels to integers
size_mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['target'] = df['target'].map(size_mapping)

# Features and target variable
X = df[['sepal l', 'sepal w', 'petal l', 'petal w']]
Y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=0)
decision_tree.fit(X_train, Y_train)

# Evaluate the model
accuracy = decision_tree.score(X_test, Y_test)
print("Accuracy of Decision Tree =", accuracy)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plt.title("Decision Tree for Iris Dataset")
plt.show()