import numpy as np
import pandas as pd

import perceptron as ppn

from sklearn.model_selection import train_test_split


state=3 #seed wymieszania


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# setosa - versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 1, -1) #nazwa danych bez znaczenia przy 2 klasach

# sepal length and petal length
X = df.iloc[0:100, [0,2]].values

perc = ppn.Perceptron(epochs=10, eta=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=state, test_size=0.20)
perc.train(X_train, y_train)
print("2 class Perceptron Setosa Versicolor:")
print(f"Number of mistakes: {(y_test != perc.predict(X_test)).sum()} out of {len(y_test)}")
print("")


#  virginica - versicolor
y2 = df.iloc[50:150, 4].values
y2 = np.where(y2 == 'Iris-virginica', 1, -1) #nazwa danych bez znaczenia przy 2 klasach

# sepal width and petal width
X2 = df.iloc[50:150, [1,3]].values
perc = ppn.Perceptron(epochs=25, eta=0.01)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=state, test_size=0.20)
perc.train(X2_train, y2_train)
print("2 class Perceptron Virginica - Versicolor:")
print(f"Number of mistakes: {(y2_test != perc.predict(X2_test)).sum()} out of {len(y2_test)}")
print("")




#  setosa - virginica
y3_1 = df.iloc[0:50, 4].values
y3_2 = df.iloc[100:150, 4].values
y3 = np.concatenate((y3_1, y3_2))
y3 = np.where(y3 == 'Iris-setosa', 1, -1) #nazwa danych bez znaczenia przy 2 klasach

# sepal width and petal width
X3_1 = df.iloc[0:50, [1,3]].values
X3_2 = df.iloc[100:150, [1,3]].values
X3 = np.concatenate((X3_1, X3_2))
perc = ppn.Perceptron(epochs=25, eta=0.01)
perc.train(X3, y3)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=state, test_size=0.20)
perc.train(X3_train, y3_train)
print("2 class Perceptron Setosa - Virginica:")
print(f"Number of mistakes: {(y3_test != perc.predict(X3_test)).sum()} out of {len(y3_test)}")
print("")


