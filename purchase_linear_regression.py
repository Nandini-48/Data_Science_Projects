import pandas as pd #dataframe -(data structure)useful for storing the data from files
import numpy as np
import matplotlib.pyplot as plt#visualization
df=pd.read_csv("ds/purchaseexcel.csv")
X=df['annual_income']
Y=df['purchase_amount']
n=len(X)
mean_x=np.mean(X)
mean_y=np.mean(Y)

numer=0
denom=0 
for i in range (n):
   numer = numer + ((X[i]-mean_x)*(Y[i]-mean_y))
   denom= denom + ((X[i]-mean_x)**2)
m= numer/denom
c=mean_y - (m*mean_x)  

max_x=np.max(X)
min_x=np.min(X)
a= np.linspace(min_x,max_x,100)
b=m*a+c
annualin=25000
pre=m*annualin+c
print("purchase amount of customer with annual income 25000rs.  is : ",pre)
plt.scatter(X,Y,color='red')
plt.title("Customer Purchasing Behaviour",fontsize=15)
plt.xlabel("Annual Income",fontsize=15)
plt.ylabel("Purchase Amount",fontsize=15)
plt.plot(a,b,color='blue')
plt.show()

