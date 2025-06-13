from sklearn import tree

# [weight, texture] - smooth 1, bumpy 0
X = [[140, 1], [130, 1], [150, 0], [170, 0]] # Features
Y = ['apple', 'apple', 'orange', 'orange'] # Labels

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print(clf.predict([[160, 0]]))
print(clf.predict([[200, 0]]))
print(clf.predict([[120, 1]]))