import numpy as np
from sklearn.neighbors import KNeighborsClassifier
weight=np.array([51,62,69,64,65,56,58,57,55])
height=np.array([167,182,176,17,172,174,169,173,170])
classes=np.array(['underweight','normal','normal','normal','normal','underweight','normal','normal','normal'])
X=np.column_stack((weight,height))
Y=classes
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X,Y)
new_sample=np.array([[57,170]])
predict_class=knn.predict(new_sample)
print(f"The predicted class for weight 57 kg and height 170 cm is : {predict_class[0]}")