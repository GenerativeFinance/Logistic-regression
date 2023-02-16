from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the iris dataset
iris = load_iris()
X = iris.data[:, :2]  # take the first two features
y = iris.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create a logistic regression model
clf = LogisticRegression(random_state=0)

# train the model on the training data
clf.fit(X_train, y_train)

# make predictions on the testing data
y_pred = clf.predict(X_test)

# calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

"""

In this example, we load the iris dataset from scikit-learn, 
which consists of measurements of iris flowers with three different 
species. We take only the first two features of the data (sepal length and width) 
and the corresponding target labels. We then split the data into a training 
set and a testing set using the train_test_split function.

Next, we create a LogisticRegression object from scikit-learn and train it 
on the training data using the fit method. We then make predictions on the 
testing data using the predict method and calculate the accuracy of the 
model using the accuracy_score function.

Note that this is a simple example and may not represent the best way to 
approach a particular dataset or problem. The scikit-learn library provides 
many additional options and tools for logistic regression and other machine learning algorithms.

"""
