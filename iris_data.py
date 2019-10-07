import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 


iris_dataset = load_iris()
print(iris_dataset.keys())
print(iris_dataset['DESCR'])

print(iris_dataset['filename'])
print(iris_dataset['target_names'])
print(iris_dataset['data'])
X_train, X_test, y_train, y_test = train_test_split(
	iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape)) 
print("y_train shape: {}".format(y_train.shape))


'''
#Vidualizing features by using scatter_matrix function from pandas lib
#features along x and y axis 
#diagonal of matrix is filled with histograms of each feature 

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                                  hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()

'''


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


prediction = knn.predict(np.array([[5, 2.9, 1, 0.2]]))
y_pred = knn.predict(X_test)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
