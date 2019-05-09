import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
import timeit


gr = 9.81
l = 1
m=1
w = (gr/l)**0.5

def row_2_step(rhs, Jy, yi, h):
    J  = Jy(yi)
    a = 1/(2+np.sqrt(2))
    B = np.eye(J.shape[0])-a*h*J

    k1 = np.linalg.solve(B, rhs(yi))
    k2 = np.linalg.solve(B, rhs(yi + 0.5*h*k1)-a*h*np.matmul(J,k1))

    return yi + h*k2


def row_3_step(rhs, Jy, yi, h):
    J  = Jy(yi)
    a = 1/(2+np.sqrt(2))
    B = np.eye(J.shape[0])-a*h*J

    k1 = np.linalg.solve(B, rhs(yi))
    k2 = np.linalg.solve(B, rhs(yi + 0.5*h*k1)-a*h*np.matmul(J,k1))
    k3 = np.linalg.solve(B, rhs(yi+h*k2) + (4+np.sqrt(2))/(2+np.sqrt(2))*h*np.matmul(J, k1)-(6+np.sqrt(2))/(2+np.sqrt(2))*h*np.matmul(J, k2))

    return yi + h/6.0*(k1 + 4*k2 +k3)


def row23(rhs, Jy, y0, T, abstol = 10**-8, h0 = None):
    tol = abstol/T
    
    if h0!=None:
        h = h0
    else:
        h = T/(1000*(np.linalg.norm(rhs(y0))+0.1))
        
    ts = [0]                #Use linked lists to make appending efficient
    ys = [y0]
    t_current = 0
    y_current = y0
    
    while t_current<T:
        r2 = row_2_step(rhs, Jy, y_current, h)
        r3 = row_3_step(rhs, Jy, y_current, h)
        
        y_current = r3
        
        t_current += h
        
        ts.append(t_current)
        ys.append(y_current)
        
        if np.linalg.norm(r3-r2)/h>tol:
            h/=2
        else:
            h*=1.1

    return np.array(ts), np.array(ys)
    
    
    
def true_lower_triangle(matrix):              #TODO: Test
    h, l = matrix.shape
    for i in range(h):
        for j in range(l):
            if matrix[i,j]!=0 and j>=i:
                return False

    return True


def explicit_rk(butcher, c, b):
    assert(true_lower_triangle(butcher))
    
    def g(rhs, y0, T, N):
        h = T/(N-1)
        solution = np.zeros((N, *y0.shape))
        solution[0] = y0
        s = butcher.shape[0]
        k = np.zeros((s, *y0.shape))
        for ti in range(N-1):
            for i in range(s):
                k[i] = rhs(h*(ti + c[i]), solution[ti]+h*np.dot(butcher[i,:i], k[:i]))
            solution[ti+1] = solution[ti]+h*np.dot(b, k)
        return solution

    return g


def implicit_rk(butcher, c, b):                             #TODO: Test
    def error_function(ks, t0, y0, dt, rhs):
        errors = np.zeros_like(ks)
        for i in range(errors.shape[0]):
            errors[i] = rhs(t0+c[i]*dt, y0+dt*np.dot(butcher[i], ks))-ks[i]

        return errors
    
    def g(rhs, y0, T, N):
        h = T/(N-1)
        solution = np.zeros((N, *y0.shape))
        solution[0] = y0
        s = butcher.shape[0]
        k = np.zeros((s, *y0.shape))
        for ti in range(N-1):
            k = nDfsolve(error_function, k, (h*ti, solution[ti], h, rhs))
            solution[ti+1] = solution[ti]+h*np.dot(b, k)
        return solution

    return g


def runge_kutta(butcher, c, b):
    for i in range(butcher.shape[0]):
        assert(np.abs(np.sum(butcher[i,:])-c[i])<0.0000001)
        
    if true_lower_triangle(butcher):
        return explicit_rk(butcher, c, b)

    return implicit_rk(butcher, c, b)

def nDfsolve(f, guess, args = ()):
    """
    Use fsolve with multi-dimensional arrays (i.e. not flattened)
    """
    original_shape = guess.shape
    return fsolve(lambda y_flat: np.ravel(f(y_flat.reshape(original_shape), *args)), np.ravel(guess)).reshape(original_shape)


def _solve_ode(f, y0, N):
    """
    Generic ODE Solver:
    f: function handle: takes previous y-value, index of time step, returns estimate of next value
    y0: first value
    N: number of steps, including first value
    """
    
    if type(y0)==np.ndarray:
        ls = np.zeros((N, *y0.shape))
    else:
        ls = np.zeros((N,1))

    ls[0] = y0
        
    for i in range(1,N):
        ls[i] = f(ls[i-1], i-1)
            
    return ls


def solve_autonomous_ode(f, y0, N, T):
    dt = float(T)/(N-1)

    if type(y0)==np.ndarray:
        ls = np.zeros((N, *y0.shape))
    else:
        ls = np.zeros((N,1))

    ls[0] = y0
        
    for i in range(1,N):
        ls[i] = f(ls[i-1], dt)
            
    return ls


"""
Definition of Evolution operators:
Consider the following autonomous ODE:

y'(t) = f(y)

An evolution operator is a function specifically tailored to f which accepts two variables, y0 == y(t) and h, where h is a float and y0 may be a float of a numpy array.
The evolution operator approximates y(t+h)
"""


def primitive_operator_factory(rhs, method="ee"):
    if method == "ee":
        return lambda y0, dt: y0+dt*rhs(y0)

    if method == "ie":
        return lambda y0, dt: fsolve(lambda approx: y0-approx+rhs(approx)*dt, y0+dt*rhs(y0))


def splitting_factory(fa, fb, method):
    if method=="SS":
        return strang_splitting(fa, fb)
    if method=="LT":
        return lie_trotter_splitting(fa, fb)
    if method=="PRKS6":
        a = np.zeros(7)
        b = np.zeros(7)
        a[1] = 0.209515106613362
        a[2] = -0.143851773179818
        a[3] = 0.5 - a[:3].sum()
        a[4:] = np.flipud(a[1:4])
        b[0] = 0.0792036964311957
        b[1] = 0.353172906049774
        b[2] = -0.0420650803577195
        b[3] = 1 - 2*b[:3].sum()
        b[4:] = np.flipud(b[:3])
        return custom_splitting_operator(fa, fb, a, b)
    if method=="Y61":
        s = 8
        a = np.zeros(s)
        b = np.zeros(s)
        a[1] = 0.78451361047755726381949763
        a[2] = 0.23557321335935813368479318
        a[3] = -1.17767998417887100694641568
        a[4] = 1.0 - 2.0*a[1:4].sum()
        a[5:] = np.flipud(a[1:4])
        b[0] = 0.5*a[1]
        b[1] = 0.5*a[1:3].sum()
        b[2] = 0.5*a[2:4].sum()
        b[3] = 0.5*(1-4*b[1]-a[3])
        b[4:] = np.flipud(b[0:4])
        return custom_splitting_operator(fa, fb, a, b)
    if method=="KL8":
        s = 18
        a = np.zeros(s)
        b = np.zeros(s)
        a[0] = 0.0
        a[1] = 0.13020248308889008087881763
        a[2] = 0.56116298177510838456196441
        a[3] = -0.38947496264484728640807860
        a[4] = 0.15884190655515560089621075
        a[5] = -0.39590389413323757733623154
        a[6] = 0.18453964097831570709183254
        a[7] = 0.25837438768632204729397911
        a[8] = 0.29501172360931029887096624
        a[9] = -0.60550853383003451169892108
        a[10:] = np.flipud(a[1:9])
        b[0:-1] = 0.5*(a[:-1]+a[1:])
        b[-1] = 1.*b[0]
        return custom_splitting_operator(fa, fb, a, b)


def custom_splitting_operator(fa, fb, ai, bi):
    """
    Produces a splitting rule (evolution operator) from the given evolution operators and coefficients
    fa: first evolution operator
    fb: second evolution operator
    ai: collection of float
    bi: collection of float

    Preconditions:
        -length of ai and bi match (add zeros yourself, if necessary)
        -sum of values in ai/bi are approximately 1
    """
    
    if abs(sum(ai)-1)>0.0001 or abs(sum(bi)-1)>0.0001:
        raise Exception("Coefficients do not add up to 1...")
    
    def operator(y0, dt):
        result = y0
        count = 0
        for a, b in zip(ai, bi):
            result = fb(fa(result, a*dt), b*dt)
        return result

    return operator


def lie_trotter_splitting(fa, fb):
    return lambda y0, dt: fb(fa(y0, dt), dt)


def strang_splitting(fa, fb):
    return lambda y0, dt: fa(fb(fa(y0, 0.5*dt), dt), 0.5*dt)


def splitted_rule(rhs, y0, T, N, evolution_operator):
    h = T/(N-1)
    return _solve_ode(lambda previous, i: evolution_operator(previous, h), y0, N)               #i is a dummy argument


def explicit_euler(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: y_previous+h*rhs(y_previous, i*h), y0, N)                                   #TODO Check


def implicit_euler(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h*rhs(y, (i+1)*h), y_previous+h*rhs(y_previous, i*h)), y0, N)        #TODO Check


def implicit_midpoint(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h*rhs(0.5*(y_previous+y), (i+0.5)*h), y_previous+h*rhs(y_previous, i*h)), y0, N)


def implicit_trapezoidal(rhs, y0, T, N):
    h = T/(N-1)
    return _solve_ode(lambda y_previous, i: fsolve(lambda y: y-y_previous-h/2*(rhs(y_previous, i*h)+rhs(y, (i+1)*h)), y_previous+h*rhs(y_previous, i*h)), y0, N)




def velocity_verlet(f, y0, v0, T, N):
    if type(y0)==np.ndarray:
        y = np.zeros((N, *y0.shape))
        v = np.zeros((N,*v0.shape))
    else:
        y = np.zeros((N,1))
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
    result = implicit_midpoint(f, np.array([0.5, 0]), 16, 20000)
    plt.plot(energy(result))
    plt.plot(result[:,0])
    plt.show()
