import numpy as np

def soft_thresholding(x, lamb):
    "Soft thresholding for the L1 norm, used for L1 regularization."
    return np.sign(x) * np.maximum(np.abs(x) - lamb, 0)
    
def steepest_sol(N, X, Y, lamb, lr=1e-4, tol=1e-6, max_iter=1000):
    '''
    N: The degree of the polynomial
    X: Independent variable data points
    Y: Dependent variable data points
    lamb: Regularization parameter (Ridge Regression)
    lr: learning
    tol: convergence criterion
    max_iter: max iteration
    '''
    A = np.zeros([len(X), N])
    b = np.array([Y]).T
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = X[i] ** (N-1-j)
    
    x0 = np.random.randn(N, 1)
    losses = []
    
    for i in range(max_iter):
        # Compute LSE & L1loss
        res = A @ x0 - b
        LSE_loss = np.sum(res**2) 
        L1_loss = lamb * np.sum(np.abs(x0))
        
        losses.append(LSE_loss + L1_loss)
        
        gradiant = 2 * A.T @ res 
        x1 = x0 - lr * gradiant
        x1 = soft_thresholding(x1, lamb * lr)
        
        if np.linalg.norm(x1-x0) < tol:
            break
        
        x0 = x1
    
    return x0.T, LSE_loss + L1_loss

# # Test
# X = np.array([1, 2, 3, 4, 5])
# Y = np.array([1.2, 2.3, 2.9, 4.1, 5.1])

# x_optimal, loss_history = steepest_sol(N=2, X=X, Y=Y, lamb=0.1)

# print(x_optimal)
# print(loss_history)