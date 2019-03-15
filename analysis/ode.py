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
    return _solve_ode(lambda y_previous, i: y_previous+h*rhs(y_previous, i*h), y0, N)                                   #TODO Check


def implicit_euler(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h*rhs(y, (i+1)*h), y_previous), y0, N)        #TODO Check


def implicit_trapezoidal(rhs, y0, T, N):
    pass


def implicit_midpoint(rhs, y0, T, N):
    pass

def velocity_verlet(f, y0, v0, T, N):
    if(type(y0)==np.ndarray):
        y = np.zeros((N, *y0.shape))
        v = np.zeros((N,*v0.shape))
    else:
        y = np.zeros((N, 1))
        v = np.zeros((N,1))
    h = T/(N-1)
    h2 = (h**2)/2.0
    y[0] = y0
    v[0] = v0
    t=0
    
    for i in range(1, N):
        y[i]=y[i-1]+h*v[i-1]+h2*f(y[i-1], t)
        v[i] = v[i-1]+h/2*(f(y[i-1], t)+f(y[i], t+h))
        t+=h

    return y,v


def f(x, t):
    return np.array([x[1], -w**2*np.sin(x[0])])

def g(x, t):
    return -w**2*np.sin(x)

if __name__=="__main__":
    plt.plot(velocity_verlet(g, np.array([0.5]), np.array([0]), 3, 20000)[0])
    plt.show()
