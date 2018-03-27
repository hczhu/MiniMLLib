import sys

import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt, matplotlib.image as mpimg

# %matplotlib inline

def plotCurves(X, Y, xlabel = '', ylabel = '', labels = []):
    assert len(X[0]) == len(Y[0])
    for i in range(len(Y)):
        plt.plot(X[i] if i < len(X) else X[0], Y[i], label = labels[i] if i < len(labels) else 'curve-{}'.format(i))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def main():
    X = list(map(float, sys.stdin.readline().strip().split(',')))
    Y, labels = [], []
    for line in sys.stdin:
        label, y = line.strip().split()
        labels.append(label)
        Y.append(list(map(float, y.split(','))))
    plotCurves([X], Y, 'wavelength', 'flux', labels)

if __name__ == "__main__":
    main()
