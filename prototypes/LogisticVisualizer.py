import sys

import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt, matplotlib.image as mpimg

# %matplotlib inline

def main():
    name = 'data/logistic_xy.txt' if len(sys.argv) < 2 else sys.argv[1]
    bluex, bluey, redx, redy = [], [], [], []
    theta = []
    with open(name, 'r') as f: 
        theta = list(map(float, f.readline().strip().split()))
        for line in f:
            x, y, label = map(float, line.strip().split())
            if label > 0: 
                bluex += [x]
                bluey += [y]
            else:
                redx += [x]
                redy += [y]
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.scatter(bluex, bluey, color='blue')
    plt.scatter(redx, redy, color='red')

    x = bluex + redx
    x.sort()
    y = []
    for xx in x:
        y.append((theta[2] + theta[0] * xx) / (-theta[1]))
    plt.plot(x, y, label = 'LR decision boundary', color = 'black')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
