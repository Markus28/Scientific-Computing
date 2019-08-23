from scicomp.linalg.decompose import factor_QR, rotation_QR
from numpy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt


def lstsq_total(A, b):
    n = A.shape[1]
    C = np.zeros((A.shape[0], A.shape[1]+1))
    C[:, :A.shape[1]] = A
    C[:, -1] = b
    print(C)
    print()
    U, S, VH = np.linalg.svd(C)
    S[n] = 0
    C_bar = np.matmul(U, np.matmul(np.diag(S), VH))
    return -1*(VH[n, :]/VH[n, n])[:-1]

def lstsq_gauss_newton(F, DF, x0, n_max = 100):
    x = x0
    n = 0
    while  n<n_max:
        s = lstsq(DF(x), -F(x))[0]
        x += s
        n+= 1
        
    return x

def lstsq_newton(phi, Dphi, Hphi, x0, n_max = 100):
    x = x0
    n = 0

    while n<n_max:
        s = np.linalg.solve(Hphi(x), Dphi(x))
        x -= s
    return x
    
def lstsq_QR(A, b):
    Q,R = factor_QR(A)
    b_bar = np.matmul(Q.T, b)[:R.shape[1]]
    R_bar = R[:R.shape[1],:]
    x = np.zeros(R.shape[1])
    
    for i in range(1, R.shape[1]+1):
        x[-i] = b_bar[-i]
        for j in range(1, i):
            x[-i]-=x[-j]*R_bar[-i, -j]
        x[-i]/=R_bar[-i, -i]

    return x

def lstsq_SVD(A, b, tol = 1e-15):
    U, S, VH = np.linalg.svd(A)
    r = 1+np.where(S/S[0]>tol)[0].max()
    return np.dot(VH[:r, :].T, np.dot(U[:,:r].T, b)/S[:r])

def lstsq_normal(A, b):
    return np.linalg.solve(np.matmul(A.T, A), np.matmul(A.T, b))
