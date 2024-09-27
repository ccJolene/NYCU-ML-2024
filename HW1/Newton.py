import numpy as np
import inverse

def Newton_sol(N, X, Y):
    '''
    N: The degree of the polynomial
    X: Independent variable data points
    Y: Dependent variable data points
    '''
    A = np.zeros([len(X), N])
    b = np.array([Y]).T
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = X[i] ** (N-1-j)

    # initialize x0
    x0 = np.random.randn(N, 1)
    Hf = 2*A.T@A
    dx = 1000
    count = 0
    
    while dx > 1e-8:
        x1 = x0 - inverse.inverse(Hf) @ (Hf@x0 - 2*A.T@b)
        
        eps = 0        
        for i in range(N):
            eps += np.abs(x1[i] - x0[i])
        
        if eps < dx:
            dx = eps
            x0 = x1
        else:
            break
        
        count += 1
        if count >= 10000:
            break
    
    loss = np.sum(np.square(A @ x0 - b))
    return x0.T, loss

# # Test
# Newton_sol(3, [1, 2, 3], [4, 5, 6])