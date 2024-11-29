import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("ds\Social_Network_Ads.csv")
from sklearn import preprocessing
le=preprocessing.LabelEncoder()#mathematical data only
df['Gender']=le.fit_transform(df['Gender'])
X=df[['Gender','Age','EstimatedSalary']]
Y=df['Purchased']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#import the class
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)  # increase the max iterations to 1000
#instantiate the model (using the defult parameters)
logreg=LogisticRegression()
#fit  the model with  data
logreg.fit(X_train,Y_train)#80% data for analysis, gives parameters
#prediction 
Y_pred=logreg.predict(X_test)#20% data used for testing
# print(Y_pred)
# print(Y_test.values)   -
#import the metrics class 
from sklearn import metrics
cnf_matrix= metrics.confusion_matrix(Y_test,Y_pred)
cnf_matrix
import seaborn as sns
class_names=[0,1]
fig,ax=plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)  
print
#create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="Greens",fmt='d',annot_kws={"size":10})
ax.xaxis.set_label_position("bottom")
plt.tight_layout
plt.title("Confusion matrix",y=1.4)
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.show()
print("Accuracy:",metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))
print("Recall:",metrics.recall_score(Y_test,Y_pred))
print("F1 Score:",metrics.f1_score(Y_test,Y_pred))
plt.scatter(df.Gender,df.Purchased,marker='+',color='red')