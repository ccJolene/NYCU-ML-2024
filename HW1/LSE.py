import numpy as np
import inverse

def LSE_sol(N, lamb, X, Y):
    '''
    N: The degree of the polynomial
    lamb: Regularization parameter (Ridge Regression)
    X: Independent variable data points
    Y: Dependent variable data points
    '''
    # Generate the design matrix A, with dimensions (len(X), N)
    A = np.zeros([len(X), N])
    b = np.array([Y]).T

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] = X[i] ** (N-1-j)
            
    # Compute the inverse of A^T A + lamb * I  
    A_reg_inv = inverse.inverse(A.T @ A + lamb * np.eye(N))
    
    # Step 3: Compute beta_hat
    beta_hat = A_reg_inv @ A.T @ b
    loss = np.sum(np.square(A @ beta_hat - b))
    
    return beta_hat, loss

# # Test
# LSE_sol(3, 0.01, [1, 2, 3], [4, 5, 6])