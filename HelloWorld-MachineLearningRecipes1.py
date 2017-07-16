from sklearn import tree


# Oranges are heavier and bumpy
# Apple are lighter and smooth

# 0 for Bumpy
# 1 for Smooth
features = [
    [140, 1], [130, 1],
    [150, 0], [170, 0]
]

# 0 for Apple
# 1 for Orange
labels = [ 0 , 0 , 1 , 1 ]

classifier = tree.DecisionTreeClassifier().fit(features, labels)

print 'Predicting item: weight 160 and bumpy'
print classifier.predict([[160, 0]])