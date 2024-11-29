# Encode categorical features
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge, Lasso
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('vehicles.csv')

# Encode categorical features
le = LabelEncoder()
df['manufacturer'] = le.fit_transform(df['manufacturer'])
df['condition'] = le.fit_transform(df['condition'])

# Scale numerical features
scaler = StandardScaler()
df[['odometer']] = scaler.fit_transform(df[['odometer']])

# Split the dataset into training and testing sets
X = df[['condition','manufacturer','odometer','year']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R-squared: {r2}')

# Analyze feature impact
print(model.coef_)

# Visualize the correlation between features
plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr(), annot=True)
plt.show()

# Feature selection using Recursive Feature Elimination (RFE)
rfe = RFE(model, 5)
rfe.fit(X_train, y_train)
print(rfe.support_)

# Regularization using Ridge Regression
ridge = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f'MSE (Ridge): {mse_ridge}, R-squared (Ridge): {r2_ridge}')

# Regularization using Lasso
lasso = Lasso(alpha=0.1, max_iter=10000).fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f'MSE (Lasso): {mse_lasso}, R-squared (Lasso): {r2_lasso}')

# Outliers detection using IQR method
Q1 = X_train['odometer'].quantile(0.25)
Q3 = X_train['odometer'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = X_train[(X_train['odometer'] < lower_bound) | (X_train['odometer'] > upper_bound)]
print(outliers)



# Scale numerical features
scaler = StandardScaler()
df[['odometer']] = scaler.fit_transform(df[['odometer']])


# Split the dataset into training and testing sets
X = df[['condition','manufacturer','odometer','year']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Evaluate the model on test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse}, R-squared: {r2}')
# Analyze feature impact
print(model.coef_)

# Visualize the correlation between features
plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr(), annot=True)
plt.show()
# Feature selection using Recursive Feature Elimination (RFE)
rfe = RFE(model, 5)
rfe.fit(X_train, y_train)
print(rfe.support_)

# Regularization using Ridge Regression
ridge = Ridge().fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f'MSE (Ridge): {mse_ridge}, R-squared (Ridge): {r2_ridge}')

# Regularization using Lasso
lasso = Lasso().fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f'MSE (Lasso): {mse_lasso}, R-squared (Lasso): {r2_lasso}')
# Outliers detection using IQR method
Q1 = X_train['odometer'].quantile(0.25)
Q3 = X_train['odometer'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = X_train[(X_train['odometer'] < lower_bound) | (X_train['odometer'] > upper_bound)]
print(outliers)

