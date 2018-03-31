#!/bin/python

def sgdL2(cof lr = None):
    x = 10
    if lr is None:
        lr = 0.5 / cof * 0.9
    while True:
        dx = 2 * cof * x
        x -= lr * dx
        print('cof={:.0f} x={:.2f} dx={:.2f}'.format(cof, x, dx))
        if abs(dx) < 1e-6: break

for cof in [1, 5, 9, 13]:
    print('Running SGD for {}x^2 with adopted learning rate'.format(cof))
    sgdL2(cof)
