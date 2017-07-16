import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

"""

print iris.feature_names
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

print iris.target_names
# ['setosa' 'versicolor' 'virginica']

print iris.data[0]
# [ 5.1  3.5  1.4  0.2]

for i in range(len(iris.target)):
    print "Example   %d: label %s, features %s" % (i+1, iris.target[i], iris.data[i])
"""

# From each type of flower, we take away 1 entry for testing
test_indices = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_indices)
train_data = np.delete(iris.data, test_indices, axis=0)

# testing data
test_target = iris.target[test_indices]
test_data = iris.data[test_indices]

classifier = tree.DecisionTreeClassifier().fit(train_data, train_target)


print test_target

# Should march test_target
print classifier.predict(test_data)