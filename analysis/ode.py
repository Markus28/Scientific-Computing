import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np

w = 3

def _solve_ode(f, y0, N):
    ls = np.zeros((N, *y0.shape))
    ls[0] = y0
        
    for i in range(1,N):
        ls[i] = f(ls[i-1], i-1)
            
    return ls


def explicit_euler(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: y_previous+h*rhs(y_previous, i*h), y0, N)                                   //TODO: Check


def implicit_euler(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h*rhs(y, (i+1)*h), y_previous), y0, N)        //TODO: Check




def f(x, t):
    return np.array([x[1], -w**2*np.sin(x[0])])

if __name__=="__main__":
    plt.plot(implicit_euler(f, np.array([0.5,0]), 3, 2000))
    plt.show()
