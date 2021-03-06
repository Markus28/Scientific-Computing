import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import numpy as np
import timeit
from scipy.linalg import expm
import sympy as sp
from numpy.polynomial.polynomial import polyval, polymul, polyint
import itertools


'''
Calculates the lagrange polynomials for the points ci
Inputs:
    - ci: List of quadrature points

Outputs:
    - The corresponding Lagrange polynomials
'''
def lagrange_polynomials(ci):
    polynomials = []
    
    for i, c in enumerate(ci):
        current_polynomial = [0,]*len(ci)
        current_polynomial[0]=1
        for j in range(len(ci)):
            if i!=j:
                current_polynomial = polymul(current_polynomial, [-ci[j]/(ci[i]-ci[j]), 1/(ci[i]-ci[j])])
        polynomials.append(current_polynomial)

    return polynomials

'''
Builds butcher tableau for collocation method
Inputs:
    - ci: List of quadrature points in [0,1], mutually distinct

Outputs:
    - A, b: Butcher tableau
'''
def collocation_to_butcher(ci):
    polynomials = lagrange_polynomials(ci)
    integrated_polynomials = [polyint(p) for p in polynomials]
    A = np.zeros((len(ci), len(ci)))
    for i, j in itertools.product(range(len(ci)), range(len(ci))):
        A[i, j] = np.polynomial.polynomial.polyval(ci[i], integrated_polynomials[j])- integrated_polynomials[j][0]

    b = np.zeros((len(ci)))
    for i in range(len(ci)):
        b[i] = polyval(1, integrated_polynomials[i]) - integrated_polynomials[i][0]
        
    return A, b




def arnoldi(A, v0, k):
    v = v0/np.linalg.norm(v0)
    V = np.zeros((v0.size, k+1))
    V[:, 0] = v
    v_bar = v
    H = np.zeros((k+1, k))

    for l in range(k):
        v_bar = np.matmul(A, V[:, l])
        for i in range(l+1):
            h = np.dot(V[:, i], v_bar)
            v_bar -= h*V[:, i]
            H[i, l] = h
        h = np.linalg.norm(v_bar)
        H[l+1, l] = h
        V[:, l+1] = v_bar/h
    return H,V

def exp_roe(rhs, J, y0, T, n):
    h = T/n
    result = np.zeros((n+1, *y0.shape))
    result[0] = y0
    y = y0.copy()
    for i in range(n):
        DF = J(y)
        y += np.dot(expm(h*DF)-np.eye(*DF.shape), np.linalg.solve(DF, rhs(y)))
        result[i+1] = y
    return result

def linear_krylov(A, y0, T, N, k):
    h = T/(N-1)
    result = np.zeros((N, *y0.shape))
    result[0] = y0
    for i in range(1, N):
        H, V = arnoldi(A, result[i-1], k)
        result[i] = np.linalg.norm(result[i-1])*np.matmul(V[:,:-1], expm(h*H[:-1]))[:,0]
    return result


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




#########################################
#                                       #
#   The following functions are for     #
#   Taylor integration in 1 Dimension   #
#                                       #
#########################################


def n_taylor_derivatives(f, n):
    derivatives = [f]
    for _ in range(n):
        expr = derivatives[-1].diff("t")+derivatives[-1].diff("x")*f
        derivatives.append(expr)

    return derivatives

def evaluate_taylor(y0, derivatives, dt):
    result = y0
    for i, der in enumerate(derivatives):
        result += np.power(dt,i+1)/np.math.factorial(i+1)*der

    return result




'''
Solve the ODE given by dx/dt = rhs(t, x) with x: R->R 1-dimensional
using a taylor method. The taylor expansion of x is computed symbolically.

Inputs:
    - rhs is a sympy expression in the variables "x" and "t"
    - y0 is x(0), a scalar
    - T is the time to which the ode should be integrated -> Solution from 0 to T
    - N is number of steps
    - n is the oder of the taylor method that should be used

Outputs:
    - (t, y) where t is the time grid and y the solution of the ode

'''
def taylor_integrate(rhs, y0, T, N, n=5):
    n = n-1
    t, h = np.linspace(0, T, N, retstep=True)
    y = np.zeros((N))
    y[0] = y0

    symbolic_derivatives = n_taylor_derivatives(rhs, n)
    function_derivatives = [sp.lambdify(["t", "x"], der, "numpy") for der in symbolic_derivatives]
    evaluate_derivatives = lambda t, x: [fd(t,x) for fd in function_derivatives]
    
    for k in range(1, N):
        y[k] = evaluate_taylor(y[k-1], evaluate_derivatives(t[k-1], y[k-1]), h)

    return t, y
