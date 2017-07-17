from sklearn import datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()

# features
X = iris.data
# labels
Y = iris.target

# test_size = .5 cause we need half the data for testing
# Example: 150 examples for Iris
# 75 will be used for training, and 75 for testing
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size= .5 )


# Using Descision Tree
from sklearn import tree
classifier = tree.DecisionTreeClassifier().fit(X_train, Y_train)

predictions = classifier.predict(X_test)
print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)


# Using KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier().fit(X_train, Y_train)

predictions = classifier.predict(X_test)
print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
