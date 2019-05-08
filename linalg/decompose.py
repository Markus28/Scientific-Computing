import numpy as np


def factor_QR(A):
    if A.size == 1:                     #Base case
        return A/np.linalg.norm(A), np.linalg.norm(A)


    x_bar = np.zeros((A.shape[1]))
    x_bar[0] = np.linalg.norm(A[:,0])

    u = A[:,0]-x_bar

    u = u/np.linalg.norm(u)
    
    mirror = (np.eye(A.shape[1])-2*np.outer(u, u))

    B = np.matmul(mirror, A)

    Q_next = np.eye((A.shape[1]))
    Q_next[1:, 1:], B[1:, 1:] = factor_QR(B[1:, 1:])

    return np.matmul(mirror.T, Q_next), B
    

def rotation_QR(A):
    if A.size == 1:                     #Base case
        return A/np.linalg.norm(A), np.linalg.norm(A)

    B = A.copy()

    Q = np.eye(B.shape[0])
    
    for i in range(1, B.shape[0]):
        if B[i, 0]!=0:
            r = np.sqrt(B[0, 0]**2 + B[i, 0]**2)
            cosine_phi = B[0, 0]/r
            sine_phi = B[i,0]/r
                
            rot = np.eye(B.shape[0])        #Build the rotation matrix
            rot[0, 0] = cosine_phi
            rot[i, 0] = -sine_phi
            rot[0, i] = sine_phi
            rot[i, i] = cosine_phi
                
            B = np.matmul(rot, B)
            Q = np.matmul(Q, rot.T)
        
    
    Q_next = np.eye(B.shape[0])
    Q_next[1:, 1:], B[1:, 1:] = rotation_QR(B[1:, 1:])    #Recurse
    
    return np.matmul(Q, Q_next), B


def gram_schmidt(A):
    Q = np.zeros_like(A, np.complex_)
    R = np.zeros_like(A, np.complex_)

    Q[:,0] = A[:, 0]
    norm = np.linalg.norm(Q[:, 0])
    Q[:, 0] = Q[:, 0]/norm
    R[0,0] = norm
    
    for i in range(1, A.shape[1]):
        coefficients = np.zeros((i), np.complex_)
        for k in range(i):
            coefficients[k] = np.dot(np.conj(A[:, i]), Q[:, k])

        Q[:, i] = A[:, i]-np.matmul(Q[:, :i], coefficients)
        norm = np.linalg.norm(Q[:, i])
        Q[:, i] = Q[:, i]/norm
        R[:i, i] = coefficients
        R[i,i] = norm

    return Q, R


def modified_gram_schmidt(A):
    R = np.zeros_like(A, np.complex_)
    Q = np.zeros_like(A, np.complex_)

    V = A.astype(np.complex_)
    
    for i in range(A.shape[1]):
        R[i,i] = np.linalg.norm(V[:, i])
        q = V[:, i]/R[i,i]
        Q[:, i] = q
        for k in range(i, A.shape[1]):
            R[i, k] = np.dot(np.conj(q), V[:, k])
            V[:, k] = V[:, k]-R[i,k]*q

    return Q, R



if __name__=="__main__":
    Q, R = factor_QR(np.array([[1.0,2.0, -4, 1], [3.0, 4.0,234.4, -3], [2,1,2,0.0004], [0,0,2,0]]))
    print(Q)
    print()
    print(R)
    print()
    print(np.matmul(Q, R))
