import numpy as np
import matplotlib.pyplot as plt

import LSE
import Newton
import Steepest

# Read the data to get X, Y
data = open('testfile.txt', 'r')
line = data.readline()

X = []
Y = []
while line:
    x, y = list(map(float, line.split(',')))
    X.append(x)
    Y.append(y)
    line = data.readline()

def plot(n, lam, case_num):
    # LSE    
    print("LSE:")
    beta_hat, LSE_loss = LSE.LSE_sol(N=n, lamb=lam, X=X, Y=Y)
    LSE_line = ''
    for i in range(n):
        if (i != 0) and (beta_hat[i][0] > 0):
            LSE_line += ' + '
        LSE_line += str(beta_hat[i][0]) + (('X^' + str(n-i-1) + '') if n-i-1 != 0 else '')
    print('Fitting Line:', LSE_line, '\nTotal error:', LSE_loss)

    # Steepest
    print("Steepest Descend:")
    Steepest_hat, Steepest_loss = Steepest.steepest_sol(N=n, X=X, Y=Y, lamb=lam)
    Steepest_line = ''
    for i in range(n):
        if (i != 0) and (Steepest_hat[0][i] > 0):
            Steepest_line += ' + '
        Steepest_line += str(Steepest_hat[0][i]) + (('X^' + str(n-i-1) + '') if n-i-1 != 0 else '')
    print('Fitting Line:', Steepest_line, '\nTotal error:', Steepest_loss)

    # Newton
    print("Newton's Method:")
    Newton_hat, Newton_loss = Newton.Newton_sol(N=n, X=X, Y=Y)
    Newton_line = ''
    for i in range(n):
        if (i != 0) and (Newton_hat[0][i] > 0):
            Newton_line += ' + '
        Newton_line += str(Newton_hat[0][i]) + (('X^' + str(n-i-1) + '') if n-i-1 != 0 else '')
    print('Fitting Line:', Newton_line, '\nTotal error:', Newton_loss)

    
    plt.figure(figsize=(6, 8))
    x_vals = np.linspace(np.floor(min(X)) - 1, np.ceil(max(X)) + 1, 250)
    y_LSE = [0]*len(x_vals)
    y_Steepest = [0]*len(x_vals)
    y_Newton = [0]*len(x_vals)
    for i in range(n):
        y_LSE += beta_hat[i][0] * x_vals ** (n-1-i)
        y_Steepest += Steepest_hat[0][i] * x_vals ** (n-1-i)
        y_Newton += Newton_hat[0][i] * x_vals ** (n-1-i)

    plt.subplot(3, 1, 1)
    plt.scatter(X, Y, color='red', s=10)
    plt.xlim((np.floor(min(X)) - 1, np.ceil(max(X)) + 1))
    plt.ylim((np.floor(min(Y)) - 1, np.ceil(max(Y)) + 1))
    y_vals = beta_hat[0][0] * x_vals + beta_hat[1][0]
    plt.plot(x_vals, y_LSE, color='black')
    plt.title('Closed-form LSE method')
    
    plt.subplot(3, 1, 2)
    plt.scatter(X, Y, color='red', s=10)
    plt.xlim((np.floor(min(X)) - 1, np.ceil(max(X))+1))
    plt.ylim((np.floor(min(Y)) - 1, np.ceil(max(Y)) + 1))
    y_vals = Newton_hat[0][0] * x_vals + Newton_hat[0][1]
    plt.plot(x_vals, y_Steepest, color='black')
    plt.title('Steepest descent method')

    plt.subplot(3, 1, 3)
    plt.scatter(X, Y, color='red', s=10)
    plt.xlim((np.floor(min(X)) - 1, np.ceil(max(X))+1))
    plt.ylim((np.floor(min(Y)) - 1, np.ceil(max(Y)) + 1))
    y_vals = Newton_hat[0][0] * x_vals + Newton_hat[0][1]
    plt.plot(x_vals, y_Newton, color='black')
    plt.title("Newton's method")

    plt.suptitle(f'Case{case_num}: n={n}, lambda={lam}')
    plt.tight_layout()
    plt.savefig(f'Case{case_num}.png')
    # plt.show()
    
# plot(2, 0, 1)
# plot(3, 0, 2)
plot(3, 10000, 3)

# Newton.Newton_sol(N=2, X=X, Y=Y)