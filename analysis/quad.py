import numpy as np
import timeit
import time
from scipy import integrate

def golub_welsch(n):
    i = np.arange(n)
    b = (i+1) / np.sqrt(4*(i+1)**2 - 1)
    J = np.diag(b, -1) + np.diag(b, 1)
    x, ev = np.linalg.eigh(J)   #x are eigenvalues
    w = 2 * ev[0,:]**2
    return x, w


def gauss(f, a, b, N):
    x,w = golub_welsch(N)
    x_transformed = 0.5*(a+b+(b-a)*x)
    values = w*f(x_transformed)
    return (b-a)/2.0 * np.sum(values)


def composite_gauss(f, a, b, n, N):     #n number of subintervals, N degree
    h = (b-a)/n
    intervals = np.linspace(a, b, n)
    x,w = golub_welsch(N)
    x_transformed = 0.5*(2*a+h+h*x)
    result = 0
    
    for i in range(n):
        result += h/2.0*np.sum(w*f(x_transformed))
        x_transformed += h
    
    return result


def mpr(f, a, b, N):
    h = float(b-a)/N
    x = np.linspace(a+h/2, b-h/2, N)
    ft = h*f(x)
    return np.sum(ft)

def tpr(f, a, b, N):
    h = float(b-a)/N
    x = np.linspace(a, b, N+1)
    return h*np.sum(f(x[1:-1]))+h/2*(f(x[0])+f(x[N]))

def simpson(f, a, b, N):
    N*=2
    h = float(b-a)/N
    x = np.linspace(a, b, N+1)
    fx = f(x)
    a = 2*(h/3)*np.sum(fx[2:-1:2])
    b = 2*(h/6)*(fx[0]+fx[-1])
    c = 8*(h/6)*np.sum(fx[1::2])
    return a+b+c


def adaptive_h(f, a, b, tol=0.00000001):                                 #Usually not efficient because of recursion. TODO: Implement in Cython
    return _adaptive_h(f,a,b,f(a),f(b), f((a+b)/2), (b-a)/6*(f(a)+4*f((a+b)/2) +f(b)),tol)


def _adaptive_h(f, a, b, fa, fb, fm, sab, tol):
    h = (b-a)/2
    fm_new1 = f(a+h/2)
    fm_new2 = f(b-h/2)
    sam = h/6*(fa+4*fm_new1+fm)
    smb = h/6*(fm+4*fm_new2+fb)

    err = np.abs(sam+smb-sab)
    
    if(err>15*tol):
        return _adaptive_h(f,a,a+h,fa,fm,fm_new1,sam,tol/2)+_adaptive_h(f,a+h,b,fm,fb,fm_new2,smb,tol/2)

    return sam+smb


def sliced(f, y):
    def g(*args):
        return f(*args, y)

    return g

def sliced_list(lf, y):
    return [sliced(f, y) for f in lf]

def quad_ruleNd(rule, n):
    """
    Generates a high dimensional quad rule from 1-dimensional blue-print

    Arguments:
    
    rule: A quadrature rule which expects:
    
        - g: a function expecting 1 argument, returning a float
        - a: float, lower bounds
        - b: float, upper bounds
        - N: resolution

    n:    Dimensions, a positive integer

    Returns a quadrature rule which expects:

        - f: A function with n float arguments, returns a float

        - bounds: A list of n functions, which return a 2-tuple of floats. The first function should expect n-1 arguments, the second n-2 etc.

        - nodes: A list of positive integers, representing the resolution along each axis

    Equivalent to:

    nquad(f, [lambda a1, a2, ... an: bounds[0](an, ... a1), ....])
    """
    
    if(n==1):
        return lambda f, bounds, nodes: rule(f, bounds[0]()[0], bounds[0]()[1], nodes[0])

    lower_rule = quad_ruleNd(rule, n-1)
    
    def f(f, bounds, nodes):
        integral = lambda y:  lower_rule(sliced(f, y), sliced_list(bounds[:-1], y), nodes[:-1])
        return rule(np.vectorize(integral), bounds[-1]()[0], bounds[-1]()[1], nodes[-1])

    return f


def cube_quad_ruleNd(rule, n):
    qnd = quad_ruleNd(rule, n)
    return lambda f, bounds, nodes: qnd(f, [lambda *args: bound for bound in bounds], nodes)







def test(f, exact_val, name):
    result = f()
    t = timeit.timeit(f, number = 30)
    print(name, ":", result, ", ERROR: ", np.abs(exact_val-result), ", TOOK:", t, "seconds")

def test_batch(f, exact):
    test(lambda: simpson(f, 0, 4, 12000),exact,"Simpson")
    test(lambda: adaptive_h(f, 0, 4),exact,"Adaptive")
    test(lambda: gauss(f, 0, 4, 80),exact,"Gauss")
    test(lambda: tpr(f, 0, 4, 120000),exact,"TPR")
    test(lambda: mpr(f, 0, 4, 120000),exact,"MPR")
    print()


def expensive(x):
    if(x<2.3234):
        return 0.0
    return x-2.3234

    
if __name__ == "__main__":
    f = lambda x: 1/(1+25*x**2)
    exact = integrate.quad(f, 0, 4)[0]

    test_batch(f, exact)

    f = lambda x: x**0.5
    exact = integrate.quad(f, 0, 4)[0]

    test_batch(f, exact)

    f = lambda x: np.sin(x)
    exact = integrate.quad(f, 0, 4)[0]

    test_batch(f, exact)

    
    exact = integrate.quad(np.vectorize(expensive), 0, 4)[0]

    test_batch(np.vectorize(expensive), exact)

