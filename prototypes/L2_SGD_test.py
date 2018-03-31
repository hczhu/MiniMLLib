#!/bin/python

def sgdL2(cof):
    x = 10
    lr = 0.5 / cof * 0.9
    while True:
        dx = 2 * cof * x
        x -= lr * dx
        print('cof={:.0f} x={:.2f} dx={:.2f}'.format(cof, x, dx))
        if abs(dx) < 1e-6: break

for cof in [1, 5, 9, 13]:
    print('Running {}x^2'.format(cof))
    sgdL2(cof)
