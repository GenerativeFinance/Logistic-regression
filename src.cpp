#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

MatrixXd gradientDescent(MatrixXd X, MatrixXd y, MatrixXd theta, int m, int n, double alpha) {
    MatrixXd h = X * theta;
    for (int i = 0; i < m; i++) {
        h(i, 0) = sigmoid(h(i, 0));
    }
    MatrixXd error = h - y;
    MatrixXd gradient = X.transpose() * error;
    gradient = gradient / m;
    theta = theta - alpha * gradient;
    return theta;
}

MatrixXd logisticRegression(MatrixXd X, MatrixXd y, double alpha, int num_iters) {
    int m = X.rows();
    int n = X.cols();
    MatrixXd theta = MatrixXd::Zero(n, 1);
    for (int i = 0; i < num_iters; i++) {
        theta = gradientDescent(X, y, theta, m, n, alpha);
    }
    return theta;
}

int main() {
    MatrixXd X(100, 3);
    X << 1, 2, 3,
         1, 2, 3,
         1, 2, 3,
         ...;
    MatrixXd y(100, 1);
    y << 0, 0, 1,
         0, 0, 1,
         0, 0, 1,
         ...;
    double alpha = 0.01;
    int num_iters = 1000;
    MatrixXd theta = logisticRegression(X, y, alpha, num_iters);
    cout << "Theta found by gradient descent: " << endl << theta << endl;
    return 0;
}
