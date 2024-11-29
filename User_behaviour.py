
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("ds/userdata.csv")

# Define features and target variable
X = df[['Device Model', 'Operating System', 'App Usage Time (min/day)', 
         'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 
         'Number of Apps Installed', 'Age', 'Gender']]
Y = df['User Behavior Class']

# Preprocess categorical variables (if any)
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, Y_train)
Y_pred_logistic = logistic_model.predict(X_test_scaled)

# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors
knn_model.fit(X_train_scaled, Y_train)
Y_pred_knn = knn_model.predict(X_test_scaled)

# Evaluate Logistic Regression Model
print("Logistic Regression Model:")
print("Accuracy:", accuracy_score(Y_test, Y_pred_logistic))
print("Classification Report:\n", classification_report(Y_test, Y_pred_logistic))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_logistic))

# Evaluate KNN Model
print("\nKNN Model:")
print("Accuracy:", accuracy_score(Y_test, Y_pred_knn))
print("Classification Report:\n", classification_report(Y_test, Y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_knn))