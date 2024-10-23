import numpy as np

################################# a. Univariate gaussian data generator ################################# 
def Random_data_Generator(mu, var):
    U = np.random.uniform(0, 1)
    V = np.random.uniform(0, 1)

    # Box-Muller Transform: Z ~ N(0, 1)
    Z = np.sqrt(-2*np.log(U)) * np.cos(2*np.pi*V)

    # X = mu + sigma*Z
    X = mu + np.sqrt(var) * Z
    
    return X

# Random_data_Generator(2, 5)

############################## b. Polynomial basis linear model data generator #############################
def Polynomial_Data_Generator(n, a, w):
    x = np.random.uniform(-1, 1)
    e = Random_data_Generator(0, a)
    
    # y = w.T * \phi(x) + e
    # phi_x = np.array([x**i for i in range(n)]).reshape(-1, 1)
    # y = np.dot(w.T, phi_x) + e
    
    y = sum([w[i] * x**i for i in range(n)]) + e
    
    return x, y

# w = np.array([1, 2, 3])
# Polynomial_Data_Generator(3, 2, w)
