import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
import scipy

def solve_linear(A, w):
    if np.isscalar(A):
        return w/A
    
    return scipy.linalg.solve(A, w)

def outer(a, b):
    if np.isscalar(a):
        return a*b

    return np.outer(a, b)
        

def newton(f, df, guess, max_n = 10, abstol = 0.000005):
    n = 0
    
    while np.linalg.norm(f(guess))>abstol and n<max_n:
        guess -= solve_linear(df(guess), f(guess))
        n+=1

    return guess

def broyden(f, df, guess, max_n = 50, abstol = 0.000005):       #Works only on arrays rn
    guess = guess.copy()
    n = 1
    f_guess = f(guess)
    J_inverse = np.linalg.inv(df)
    s = np.matmul(J_inverse, f_guess)
    guess -= s
    f_guess = f(guess)
    
    while np.linalg.norm(f_guess)>abstol and n<max_n:
        w = np.matmul(J_inverse, f_guess)
        s_norm = np.linalg.norm(s)**2
        J_inverse += np.matmul(np.outer(w, s), J_inverse)/(s_norm-np.dot(s,w))
        z = np.dot(s, w)
        s = (1+z/(s_norm-z))*w
        guess -= s
        f_guess = f(guess)
        n+=1

    return guess



def secant(f, x0, x1, max_n = 50, abstol = 0.000005):
    n = 0
    
    while np.linalg.norm(f(x1))>abstol and n<max_n:
        df = (f(x1)-f(x0))/(x1-x0)
        tmp = x1
        x1 -= 1.0/df*f(x1)
        x0 = tmp
        n+=1

    return x1



