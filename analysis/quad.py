import numpy as np
from scipy import integrate

def mpr(f, a, b, N):
    h = float(b-a)/N
    x = np.linspace(a+h, b-h, N)
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




def _golub_welsch(n):
    i = np.arange(n)
    b = (i+1) / np.sqrt(4*(i+1)**2 - 1)
    J = np.diag(b, -1) + np.diag(b, 1)
    x, ev = np.linalg.eigh(J)   #x are eigenvalues
    w = 2 * ev[0,:]**2
    return x, w


def gauss(f, a, b, n):
    x,w = _golub_welsch(n)
    x_bar = 0.5*(a+b+(b-a)*x)
    values = w*f(x_bar)
    return (b-a)/2.0 * np.sum(values)


if __name__ == "__main__":
    print("GROUND TRUTH: ",integrate.quad(lambda x: 1/(1+25*x**2), 0, 4)[0])
    print("Simpson: ", simpson(lambda x: 1/(1+25*x**2), 0, 4, 120000))
    print("TPR: ",tpr(lambda x: 1/(1+25*x**2), 0, 4, 10000))
    print("MPR: ",mpr(lambda x: 1/(1+25*x**2), 0, 4, 50000))
    print("GAUSS: ", gauss(lambda x: 1/(1+25*x**2), 0, 4, 80))

    print()

    print("GROUND TRUTH: ",integrate.quad(lambda x: x**0.5, 0, 4)[0])
    print("Simpson: ",simpson(lambda x: x**0.5, 0, 4, 120000))
    print("TPR: ",tpr(lambda x: x**0.5, 0, 4, 10000))
    print("MPR: ",mpr(lambda x: x**0.5, 0, 4, 50000))
    print("GAUSS: ",gauss(lambda x: x**0.5, 0, 4, 80))

    print()

    print("GROUND TRUTH: ",integrate.quad(lambda x: np.sin(x), 0, 4)[0])
    print("Simpson: ",simpson(lambda x: np.sin(x), 0, 4, 120000))
    print("TPR: ",tpr(lambda x: np.sin(x), 0, 4, 10000))
    print("MPR: ",mpr(lambda x: np.sin(x), 0, 4, 50000))
    print("GAUSS: ",gauss(lambda x: np.sin(x), 0, 4, 80))

