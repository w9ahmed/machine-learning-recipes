from scipy.spatial import distance

def euc(a, b):
    distance.euclidean(a, b)

class ScrappyKNN():
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        return

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.Y_train[best_index]

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


classifier = ScrappyKNN()
classifier.fit(X_train, Y_train)

predictions = classifier.predict(X_test)
print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
