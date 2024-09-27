import numpy as np

def inverse(A):
    '''
    Use LU decomposision to find the inverse matrix of A
    '''
    dim = A.shape[0]
    L, U = LUdecomposision(A)
    Y = np.eye(dim)
    
    # Forward substitution to solve L * Y = I
    for i in range(1, dim):
        for j in range(i):
            y_val = 0
            for k in range(i):
                y_val += L[i][k] * Y[k][j]
            Y[i][j] = -y_val
            
    # Backward substitution to solve U * A_inverse = Y
    A_inv = np.zeros([dim, dim])
    for i in range(dim-1, -1, -1):
        for j in range(dim):
            A_inv_val = 0
            for k in range(i+1, dim):
                A_inv_val += U[i][k] * A_inv[k][j] 
            A_inv[i][j] = (Y[i][j] - A_inv_val) / U[i][i]
            
    return A_inv
    

def LUdecomposision(A):
    '''
    initialize:
        L: Identical matrix
        U: Zero matrix has same shape with A
        
    return L, U
    '''
    dim = A.shape[0]
    L = np.zeros([dim, dim])
    U = np.zeros([dim, dim])
    for i in range(dim):
        # L
        for j in range(i):
            L[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(j))) / U[j][j]        
        L[i][i] = 1
        
        # U
        for j in range(i, dim):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
            
    return L, U


# # Test
# A = np.array([[4, 3], [3, 2]])
# A_inv = inverse(A)

# print("A Inverse:\n")
# print(A_inv)

# print("\nCheck if A * A_inv is identical matrux")
# print(np.dot(A,  A_inv))
