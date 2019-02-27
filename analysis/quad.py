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

def cross(f, y):
    def g(*args):
        return f(*args, y)

    return g

    
def quad_ruleNd(rule, n):
    if(n==1):
        return lambda f, bounds, nodes: rule(f, bounds[0][0], bounds[0][1], nodes[0])
    
    def f(f, bounds, nodes):
        lower_rule = quad_ruleNd(rule, n-1)
        integral = lambda y:  lower_rule(cross(f, y), bounds[:-1], nodes[:-1])
        return rule(np.vectorize(integral), bounds[-1][0], bounds[-1][1], nodes[-1])

    return f


if __name__ == "__main__":
    print("GROUND TRUTH: ",integrate.quad(lambda x: 1/(1+25*x**2), 0, 4)[0])
    print("Simpson: ", simpson(lambda x: 1/(1+25*x**2), 0, 4, 120000))
    print("TPR: ",tpr(lambda x: 1/(1+25*x**2), 0, 4, 10000))
    print("MPR: ",mpr(lambda x: 1/(1+25*x**2), 0, 4, 50000))

    print()

    print("GROUND TRUTH: ",integrate.quad(lambda x: x**0.5, 0, 4)[0])
    print("Simpson: ",simpson(lambda x: x**0.5, 0, 4, 120000))
    print("TPR: ",tpr(lambda x: x**0.5, 0, 4, 10000))
    print("MPR: ",mpr(lambda x: x**0.5, 0, 4, 50000))

    print()

    print("GROUND TRUTH: ",integrate.quad(lambda x: np.sin(x), 0, 4)[0])
    print("Simpson: ",simpson(lambda x: np.sin(x), 0, 4, 120000))
    print("TPR: ",tpr(lambda x: np.sin(x), 0, 4, 10000))
    print("MPR: ",mpr(lambda x: np.sin(x), 0, 4, 50000))

