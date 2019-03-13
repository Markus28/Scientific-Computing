import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np

w = 3

def explicit_euler(f, y0, T, N):
    ls = np.zeros((N, *y0.shape))
    ls[0] = y0
    h = T/N
    x = 0
    
    for i in range(1,N):
        ls[i] = ls[i-1]+h*f(ls[i-1], x)
        x+=h

    return ls


def implicit_euler(f, y0, T, N):
    ls = np.zeros((N, *y0.shape))
    ls[0] = y0
    h = T/N
    x = 0
    for i in range(1,N):
        ls[i] = fsolve(lambda y: y-ls[i-1]-h*f(y, x), ls[i-1])
        x+=h

    return ls


def f(x, t):
    return np.array([x[1], -w**2*np.sin(x[0])])

if __name__=="__main__":
    plt.plot(explicit_euler(f, np.array([0.5, 0]), 6, 2500))
    plt.show()
