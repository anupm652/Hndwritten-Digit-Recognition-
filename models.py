#import necessary libraries
import pandas as pd
import numpy as np
import joblib

#importing the appropriate model
from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=3)

#Data Preprocessing
df=pd.read_csv("train.csv")
X=np.c_[df.drop("label",axis=1)]
y=np.c_[df["label"]]

#Train the model on the data
knn_clf.fit(X,y)

#Save the model
joblib.dump(knn_clf,"hrdr.pkl")

print("Task done")