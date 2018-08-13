
import pandas as pd 
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



cancer=load_breast_cancer()

#print(cancer.keys())

X= pd.DataFrame(cancer.data)
y=pd.DataFrame(cancer.target)


X.columns = cancer.feature_names
y.columns = ['target']

#print(X.head())
#print(y.head())

df = pd.concat([X,y], axis=1)
X_new= df.loc[: , 'mean radius': 'worst fractal dimension']
y_new = df.loc[:, 'target']
#Creating train and test set

X_train, X_test, y_train, y_test = train_test_split(X_new,y_new, random_state=0)

#knn with n=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

y_pred= knn.predict(X_test)


print('recall score for n_neighbour=1 is:' , metrics.recall_score(y_test, y_pred))
print('precision score for n_neighbour=1 is:' , metrics.precision_score(y_test, y_pred))
print ('f1 score for n_neighbour =1 is:', metrics.f1_score(y_test,y_pred))
print('score for n_neighbour=1 is:', knn.score(X_test,y_test))



