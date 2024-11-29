import pandas as pd #dataframe -(data structure)useful for storing the data from files
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
df=pd.read_csv("iris.data.csv")
size_mapping = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
df['target'] = df['target'].map(size_mapping)
X=df[['sepal l',	'sepal w',	'petal l',	'petal w']]
Y=df['target']

scaler = StandardScaler()
x_scaler= scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#import the class
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
accuracy = knn.score(X_test, Y_test)
print("Accuracy of knn =",accuracy)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()
#fit  the model with  data
logreg.fit(X_train,Y_train)#80% data for analysis, gives parameters
#prediction 
Y_pred=logreg.predict(X_test)
from sklearn import metrics
cnf_matrix= metrics.confusion_matrix(Y_test,Y_pred)
cnf_matrix
print(cnf_matrix)
print("Accuracy of logistic reression:",metrics.accuracy_score(Y_test,Y_pred))

