class LogisticRegression {
  constructor(numFeatures) {
    this.weights = new Array(numFeatures).fill(0);
    this.bias = 0;
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
  }

  predict(features) {
    const z = this.bias + features.reduce((acc, val, i) => acc + val * this.weights[i], 0);
    return this.sigmoid(z);
  }

  train(features, labels, learningRate, numIterations) {
    for (let i = 0; i < numIterations; i++) {
      let errorSum = 0;
      for (let j = 0; j < features.length; j++) {
        const predicted = this.predict(features[j]);
        const error = labels[j] - predicted;
        errorSum += error;
        this.weights = this.weights.map((w, k) => w + error * features[j][k] * learningRate);
        this.bias += error * learningRate;
      }
      if (errorSum === 0) break;
    }
  }
}

// example usage
const X = [[1, 2], [2, 3], [3, 4], [4, 5]];
const y = [0, 0, 1, 1];
const clf = new LogisticRegression(X[0].length);
clf.train(X, y, 0.1, 1000);
console.log(clf.predict([1, 1])); // should output a low value close to 0
console.log(clf.predict([4, 6])); // should output a high value close to 1


/*
In this example, we define a LogisticRegression class with methods for calculating 
the sigmoid function, making predictions, and training the model. The predict 
method takes an array of feature values and returns a probability estimate between 
0 and 1. The train method takes arrays of feature values and corresponding labels, 
along with hyperparameters for the learning rate and number of iterations.

The implementation uses stochastic gradient descent to update the model parameters 
for each training example. The weights array and bias value are updated using the 
error between the predicted and actual label values. The train method repeats this 
process for the specified number of iterations, or until the total error reaches 0.

The example usage code creates a simple dataset of four points with two features and 
binary labels. It trains a logistic regression model on this dataset and uses the 
predict method to make predictions for two new points. The results should be 
consistent with the expected behavior of a logistic regression model.
*/
