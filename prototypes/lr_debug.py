from __future__ import division
import numpy as np

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def log_loss(X, Y, theta):
    margins = Y * X.dot(theta)
    return np.sum(-np.log(1. / (1 + np.exp(-margins)))) / X.shape[0]

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    while i < 100:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        print(grad)
        theta = theta  - learning_rate * (grad)
        if i % 1 == 0:
            print('Finished %d iterations logloss = %.3f'%(i, log_loss(X, Y, theta)))
            # print('Finished %d iterations with diff norm = %.16f'%(i, np.linalg.norm(prev_theta - theta)))
        if np.linalg.norm(prev_theta - theta) < 1e-12:
            print('Converged in %d iterations' % i)
            break
    return

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = load_data('data/data_a.txt')
    logistic_regression(Xa, Ya)

    print('\n==== Training model on data set B ====')
    # Xb, Yb = load_data('data/data_b.txt')
    # logistic_regression(Xb, Yb)

    return

if __name__ == '__main__':
    main()
