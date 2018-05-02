import numpy as np
# import matplotlib.pyplot as plt
import sys
from datetime import datetime

def readData(csv_file, has_labels = True):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    if not has_labels:
        return data, None
    x = data[:, 1:]
    y = data[:, 0]
    return x, y

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    Column wise
    """
    x = np.exp(x - np.max(x, 0))
    return x / x.sum(0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.clip(-100, 100)))

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def nn_train(trainData, trainLabels, devData, devLabels, testData):
    (m, n) = trainData.shape
    O = trainLabels.shape[1]
    num_hidden = 300
    learning_rate = 5
    B = 256
    W1 = np.random.randn(num_hidden, n)
    B1 = np.zeros(num_hidden, dtype=float)

    W2 = np.random.randn(O, num_hidden)
    B2 = np.zeros(O, dtype=float)
    L = 0.0001
    def forward(x, y):
        z1 = np.matmul(W1, x) + B1[:, np.newaxis]
        a1 = sigmoid(z1)
        z2 = np.matmul(W2, a1) + B2[:, np.newaxis]
        a2 = softmax(z2)
        pred = np.argmax(a2, 0)
        accuracy = 1.0 * np.sum((pred == np.argmax(y, 0)).astype(int)) / y.shape[1]
        xen = -np.sum(y * np.log(a2 + 1e-20)) / y.shape[1]
        return z1, a1, z2, a2, xen, accuracy, pred
    def report(name, x, y):
        _, _, _, _, xen, accuracy, _ = forward(x.T, y.T)
        sys.stderr.write('{}: CEL = {:.12f} accuracy = {:.4f}\n'.format(
            name,
            xen,
            accuracy)) 
        sys.stderr.flush()
        return (xen, accuracy)
    E = 100
    train_log_loss = 0
    dev_acc_stop_thrshold = 0.97
    for epoch in range(1, E + 1):
        for b in range(0, m, B):
            # Reference: http://cs229.stanford.edu/notes/cs229-notes-backprop.pdf
            a0 = trainData[b:(b + B), :].T
            S = a0.shape[1]
            y = trainLabels[b:(b + B), :].T
            z1, a1, z2, a2, xen, _, _ = forward(a0, y)
            dz2 = a2 - y
            dw2 = np.matmul(dz2, a1.T)

            d_a1_z1 = a1 * (1 - a1)
            dz1 = np.matmul(W2.T, dz2) * d_a1_z1
            dw1 = np.matmul(dz1, a0.T)

            W2 -= (learning_rate / S) * dw2 + (2 * L * learning_rate * W2)
            B2 -= (learning_rate / S) * dz2.sum(1)
            W1 -= (learning_rate / S) * dw1 + (2 * L * learning_rate * W1)
            B1 -= (learning_rate / S) * dz1.sum(1)
            train_log_loss = train_log_loss * 0.9 + 0.1 * xen
            if b + B >= m:
                sys.stderr.write('Smoothed training log-loss: {:.4f}\n'.format(train_log_loss))
        report('Dev at epoch #{}'.format(epoch), devData, devLabels)
        learning_rate *= 0.98
    report('Train', trainData, trainLabels)
    _, _, _, _, _, _, pred = forward(testData.T, one_hot_labels(np.zeros(testData.shape[0])).T)
    print('ImageId,Label')
    for i in range(pred.size):
        print('{},{}'.format(i + 1, pred[i]))            

def main():
    begin = datetime.now()
    np.random.seed(135)
    suffix = ''
    trainData, trainLabels = readData('mnist-data/train.csv')
    trainLabels = one_hot_labels(trainLabels)
    N = trainData.shape[0]
    DN = 2000
    p = np.random.permutation(N)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:DN,:]
    devLabels = trainLabels[0:DN,:]
    trainData = trainData[DN:N,:]
    trainLabels = trainLabels[DN:N,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, _ = readData('mnist-data/test.csv', False)
    testData = (testData - mean) / std

    sys.stderr.write('Loaded all data.\n')
    sys.stderr.flush()
    datatime = datetime.now()
    sys.stderr.write('Data loading time = {:d} seconds\n'.format((datatime - begin).seconds))
    sys.stderr.flush()
    nn_train(trainData, trainLabels, devData, devLabels, testData)
    sys.stderr.write('Training time = {:d} seconds\n'.format((datetime.now() - datatime).seconds))
    sys.stderr.flush()

if __name__ == '__main__':
    main()
