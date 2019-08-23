import numpy as np
from scipy.linalg import lu_factor, lu_solve, expm
import matplotlib.pyplot as plt


def eig_qr(A, max_n = 100, tol=1e-10):
    d = A.shape[0]
    n = 0
    M = A.copy()

    while n<max_n and np.linalg.norm(np.tril(M, -1))**2/A.size>tol:
        ew,ev = np.linalg.eig(M[-2:,-2:])
        k = np.argmin(ew-M[-1,-1])
        shift_value = ew[k]
        Q, R = np.linalg.qr(M-shift_value*np.eye(d))
        M = np.matmul(Q.T, np.matmul(M, Q))
        n+= 1

    return M.diagonal()
    
def largest_eval(A, n_max = 500):
    z = np.random.rand((A.shape[0]))
    z /= np.linalg.norm(z)
    n = 0
    
    while n<n_max:
        w = np.matmul(A, z)
        z = w/np.linalg.norm(w)
        n+=1

    return z, np.dot(np.conj(z), np.matmul(A, z))

def smallest_eval(A, n_max = 500):
    LUP = lu_factor(A)
    n = 0
    z = np.random.rand((A.shape[0]))
    z /= np.linalg.norm(z)

    while n<n_max:
        w = lu_solve(LUP, z)
        z = w/np.linalg.norm(w)
        n += 1

    return z, np.dot(np.conj(z), np.matmul(A, z))
    

def closest_eval(A, ev, n_max = 500,):
    z, e = smallest_eval(A-ev*np.eye(A.shape[0]), n_max)
    return z, ev+e


def closest_eval_shifted(A, ev, n_max = 500, k = 10):
    n = 0
    z = np.random.rand((A.shape[0]))
    z /= np.linalg.norm(z)
    while n<n_max:
        if n%k == 0:
            if n != 0:
                ev = np.dot(np.conj(z), np.matmul(A, z))
                M = A-ev*np.eye(A.shape[0])
            else:
                M = A-ev*np.eye(A.shape[0])
                
            LUP = lu_factor(M)

        w = lu_solve(LUP, z)
        z = w/np.linalg.norm(w)
        n += 1

    return z, np.dot(np.conj(z), np.matmul(A, z))

def arnoldi(A, v0, k):
    v = v0/np.linalg.norm(v0)
    V = np.zeros((v0.size, k+1))
    V[:, 0] = v
    v_bar = v
    H = np.zeros((k+1, k))
    
    for l in range(k):
        v_bar = np.matmul(A, V[:,l])
        for i in range(l+1):
            h = np.dot(V[:, i], v_bar)
            v_bar -= h*V[:, i]
            H[i, l] = h
        h = np.linalg.norm(v_bar)
        H[l+1, l] = h
        V[:, l+1] = v_bar/h

    return H, V


def lanczos(A, v0, k):
    v = v0/np.linalg.norm(v0)
    V = np.zeros((v0.size, k+1))
    V[:, 0] = v
    v_bar = v
    H = np.zeros((k+1, k))
    
    for l in range(k):
        v_bar = np.matmul(A, V[:,l])
        for i in range(l-1, l+1):
            h = np.dot(V[:, i], v_bar)
            v_bar -= h*V[:, i]
            H[i, l] = h
        h = np.linalg.norm(v_bar)
        H[l+1, l] = h
        V[:, l+1] = v_bar/h

    return H, V
    

