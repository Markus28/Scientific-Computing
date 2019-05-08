import numpy as np


def solve_QR(Q, R, y):
    #Solve QR x = y

    w = np.matmul(Q.T, y)
    
    x = np.zeros((R.shape[1]))

    for i in range(1, x.size+1):
        x[-i] = w[-i]
        for j in range(1,i):
            x[-i] -= R[-i, -j]*x[-j]

        x[-i]/=R[-i, -i]

    return x


if __name__=="__main__":
    from decompose import factor_QR, rotation_QR
    A = np.array([[-1, -2, 1], [-1, -3, -2], [2,1,-3]])
    Q, R = factor_QR(A)
    print(np.linalg.solve(A, np.array([0,0,0])))
    print(solve_QR(Q, R, np.array([0,0,0])))
