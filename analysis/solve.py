import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
import scipy

def lu_factor(A):
    if np.isscalar(A):
        return A

    return scipy.linalg.lu_factor(A)

def lu_solve(lup, b):
    if np.isscalar(lup):
        return b/lup

    return scipy.linalg.lu_solve(lup, b)

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


def broyden(f, df, guess, max_n = 50, abstol = 0.000005):
    n = 1
    
    if np.isscalar(guess):
        previous_s = np.zeros((max_n))
    else:
        previous_s = np.zeros((max_n, *guess.shape))
        
    previous_s_abs = np.zeros((max_n))
    
    lup = lu_factor(df)
    s = lu_solve(lup, f(guess))
    guess -= s

    previous_s[0] = s
    previous_s_abs[0] = np.dot(s,s)

    
    while np.linalg.norm(f(guess))>abstol and n<max_n:
        w = lu_solve(lup, f(guess))
        for k in range(1, n):
            w += previous_s[n]*(np.dot(previous_s[n-1], w))/previous_s_abs[n-1]

        z = np.dot(s, w)
        s = (1+z/(previous_s_abs[n-1]-z))*w
        previous_s_abs[n] = np.dot(s,s)
        previous_s[n] = s
        guess -= s
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


def test():
    f = lambda x: np.arctan(x)+x
    df = lambda x: 1+1/(x**2+1)
    methods = [newton, broyden, secant]
    names = ["newton", "broyden", "secant"]
    args = [(f, df, 100.0), (f, 100.0), (f, 100.0,1.1)]
    steps = [1,3,4,5,6,7,8,9,10]
    for method, arg, name in zip(methods, args, names):
        errors = []
        for step in steps:
            errors.append(abs(f(method(*arg, max_n=step, abstol=0.00000000000000005))))

        plt.plot(steps, [np.log(e) for e in errors])

        print(errors)
        
    plt.legend(names)
    plt.show()
            

def test_b():
    F = lambda x: np.array([x[0]**2-x[1]**4, x[0]-x[1]**3])
    dF = lambda x: np.array([[2*x[0], -4*x[1]**3],[1, -3*x[1]**2]])

    guess = np.array([0.7,0.7])
    steps = [1,2,3,4,5,6,7,8,9,10,11]
    errors = []

    for step in steps:
        errors.append(np.linalg.norm(newton(F, dF, guess, max_n=step, abstol=10**-15)-np.array([1,1])))

    print(errors)
    plt.plot(steps, [np.log10(e) for e in errors])

    
    guess = np.array([0.7,0.7])

    errors = []
    for step in steps:
        errors.append(np.linalg.norm(broyden(F, dF(guess), guess, max_n=step, abstol=10**-25)-np.array([1,1])))

    print(errors)
    plt.plot(steps, [np.log10(e) for e in errors])
    
    plt.legend(["newton", "broyden"])
    plt.show()


    n = 5
    h = 2.0/n
    guess = np.linspace(2, 4-h, n)
    b = np.linspace(1,n,n)
    print(b)
    a = 1.0/(np.sqrt(np.sum(b)-1))*(b-1)
    A = np.eye(n)+np.outer(a,a)
    F = lambda x: np.matmul(np.diag(x), np.matmul(A, x))-b

    print(F(guess))

    steps = [1,2,3,4,5,6,7,8,9,10,11]
    errors = []

    for step in steps:
        errors.append(np.linalg.norm(F(newton(F, dF, guess, max_n=step, abstol=10**-15))))

    print(errors)
    plt.plot(steps, [np.log10(e) for e in errors])

    
    guess = np.linspace(2, 4-h, n)

    errors = []
    for step in steps:
        errors.append(np.linalg.norm(F(broyden(F, dF(guess), guess, max_n=step, abstol=10**-25))))

    print(errors)
    plt.plot(steps, [np.log10(e) for e in errors])
    
    plt.legend(["newton", "broyden"])
    plt.show()

    
if __name__=="__main__":
    test_b()
