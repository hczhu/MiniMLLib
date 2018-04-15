import numpy as np
import matplotlib.pyplot as plt
import sys

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
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
    """
    Compute the sigmoid function for the input here.
    """
    return 1.0 / (1 + np.exp(-x.clip(-100, 100)))

def nn_train(trainData, trainLabels, devData, devLabels, testData, testLabels):
    (m, n) = trainData.shape
    O = trainLabels.shape[1]
    num_hidden = 300
    learning_rate = 5
    B = 1000
    W1 = np.random.randn(num_hidden, n)
    B1 = np.zeros(num_hidden)

    W2 = np.random.randn(O, num_hidden)
    B2 = np.zeros(O)
    L = 0.0001
    def forward(x, y):
        z1 = np.matmul(W1, x) + B1[:, np.newaxis]
        a1 = sigmoid(z1)
        z2 = np.matmul(W2, a1) + B2[:, np.newaxis]
        a2 = softmax(z2)
        accuracy = 1.0 * np.sum((np.argmax(a2, 0) == np.argmax(y, 0)).astype(int)) / y.shape[1]
        xen = -np.sum(np.log(a2 * y + 1e-20)) / y.shape[1]
        return z1, a1, z2, a2, xen, accuracy
    def report(name, x, y):
        _, _, _, _, xen, accuracy = forward(x.T, y.T)
        sys.stderr.write('{}: CEL = {:.12f} accuracy = {:.4f}\n'.format(
            name,
            xen,
            accuracy)) 
        sys.stderr.flush()
        return (xen, accuracy)
    dev_stats = []
    train_stats = []
    E = 30
    print('Epoch {}'.format(','.join([str(e) for e in range(E)])))
    for epoch in range(1, E + 1):
        for b in range(0, m, B):
            # Reference: http://cs229.stanford.edu/notes/cs229-notes-backprop.pdf
            a0 = trainData[b:(b + B), :].T
            S = a0.shape[1]
            y = trainLabels[b:(b + B), :].T
            z1, a1, z2, a2, xen, _ = forward(a0, y)
            dz2 = a2 - y
            dw2 = np.matmul(dz2, a1.T)

            d_a1_z1 = a1 * (1 - a1)
            dz1 = np.matmul(W2.T, dz2) * d_a1_z1
            dw1 = np.matmul(dz1, a0.T)

            W2 -= (learning_rate / S) * dw2 + ((2.0 * L ) * W2)
            B2 -= (learning_rate / S) * dz2.sum(1)
            W1 -= (learning_rate / S) * dw1 + ((2.0 * L) * W1)
            B1 -= (learning_rate / S) * dz1.sum(1)
        dev_stats.append(report('dev at epoch #{}'.format(epoch), devData, devLabels))
        train_stats.append(report('train at epoch #{}'.format(epoch), trainData, trainLabels))

    print('train-CEL {}'.format(','.join(['{:.2f}'.format(x[0]) for x in train_stats])))
    print('dev-CEL {}'.format(','.join(['{:.2f}'.format(x[0]) for x in dev_stats])))

    print('dev-accuracy {}'.format(','.join(['{:.4f}'.format(x[1]) for x in dev_stats])))
    print('train-accuracy {}'.format(','.join(['{:.4f}'.format(x[1]) for x in train_stats])))
    report('Test', testData, testLabels)
            
def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    suffix = ''
    trainData, trainLabels = readData('data/images_train.csv' + suffix, 'data/labels_train.csv' + suffix)
    trainLabels = one_hot_labels(trainLabels)
    N = 60000
    DN = 10000
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

    testData, testLabels = readData('data/images_test.csv', 'data/labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
    sys.stderr.write('loaded all data.\n')
    sys.stderr.flush()
    nn_train(trainData, trainLabels, devData, devLabels, testData, testLabels)

if __name__ == '__main__':
    main()
