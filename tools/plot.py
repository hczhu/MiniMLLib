#!/bin/python
import sys

import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt, matplotlib.image as mpimg
import argparse
import collections

# %matplotlib inline

def plotCurves():
    """
        The input example:
        x_label x[0],x[1],x[2], ...
        y1_label y1[0],y2[0],...
        y2_label y2[0],y2[0],...
        y3_label y3[0],y2[0],...
        ...
    """
    xLabel, Xs = sys.stdin.readline().strip().split()
    X = list(map(float, Xs.split(',')))
    sortedXIdx = list(range(len(X)))
    sortedXIdx.sort(key = lambda a: X[a])
    X = [X[sortedXIdx[i]] for i in range(len(sortedXIdx))]
    Y, yLabels = [], []
    for line in sys.stdin:
        label, y = line.strip().split()
        yLabels.append(label)
        Y.append(list(map(float, y.split(','))))
    for i in range(len(Y)):
        assert(len(Y[i]) == len(X))
        Y[i] = [Y[i][sortedXIdx[j]] for j in range(len(sortedXIdx))]
        plt.plot(X, Y[i], label = yLabels[i])

    plt.xlabel(xLabel)
    plt.legend()
    plt.grid()
    plt.show()

def plotScatters():
    """
        The input example:
        x_label y_label
        label1 x1,y1
        label2 x2,y2
        ...
    """
    kColors = [
        'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w',
    ]
    xLabel, yLabel = sys.stdin.readline().strip().split()
    labelToPoints = collections.defaultdict(list)
    labelToIndex = {}
    for line in sys.stdin:
        label, xy = line.strip().split()
        x, y = list(map(float, xy.split(',')))
        labelToPoints[label].append((x, y))
        if label not in labelToIndex:
            labelToIndex[label] = len(labelToPoints) - 1

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    for label, points in labelToPoints.items():
        plt.scatter([xy[0] for xy in points],
                    [xy[1] for xy in points],
                    label = label,
                    color=kColors[labelToIndex[label]] if labelToIndex[label] < len(kColors)
                        else np.random.uniform(0, 1.0, 3))
    plt.grid()
    plt.legend()
    plt.show()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'scatter':
        plotScatters()
    else:
        plotCurves()

def test():
    sys.stdin = open('data/scatter_test_data.txt', 'r')
    sys.argv.append('scatter')
    main()
    sys.stdin = open('data/curve_test_data.txt', 'r')
    sys.argv[-1] = 'curve'
    sys.argv.append('1')
    main()

if __name__ == "__main__":
   if sys.argv[-1] == 'test':
        sys.argv = sys.argv[0:-1]
        test()
   else:
        main()
