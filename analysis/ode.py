import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
import timeit

gr = 9.81
l = 1
m=1
w = (gr/l)**0.5

def _solve_ode(f, y0, N):
    if type(y0)==np.ndarray:
        ls = np.zeros((N, *y0.shape))
    else:
        ls = np.zeros((N,1))
    ls[0] = y0
        
    for i in range(1,N):
        ls[i] = f(ls[i-1], i-1)
            
    return ls



def explicit_euler(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: y_previous+h*rhs(y_previous, i*h), y0, N)                                   #TODO Check


def implicit_euler(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h*rhs(y, (i+1)*h), y_previous+h*rhs(y_previous, i*h)), y0, N)        #TODO Check


def implicit_trapezoidal(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h*rhs(0.5*(y_previous+y), (i+0.5)*h), y_previous+h*rhs(y_previous, i*h)), y0, N)


def implicit_midpoint(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h/2*(rhs(y_previous, i*h)+rhs(y, (i+1)*h)), y_previous+h*rhs(y_previous, i*h)), y0, N)

def velocity_verlet(f, y0, v0, T, N):
    if type(y0)==np.ndarray:
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

def energy(x):
    return (1-np.cos(x[:,0]))*m*gr*l + 0.5*m*(x[:,1]**2)

if __name__=="__main__":
    result = implicit_trapezoidal(f, np.array([0.5, 0]), 16, 2000)
    plt.plot(energy(result))
    plt.plot(result[:,0])
    plt.show()
