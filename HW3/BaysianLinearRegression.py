from RandomDataGenerator import Polynomial_Data_Generator
import numpy as np
import matplotlib.pyplot as plt

def Baysian_Linear_Regression(b, n, a, w):
    i, max_iter = 0, 1000
    point_x, point_y = [], []
    visual_x, visual_y = [], []
    pred_x, pred_y = [], []
    mu = np.zeros((n, 1))
    var = 1/b * np.identity(n)   
    
    while i < max_iter:
        
        x, y = Polynomial_Data_Generator(n, a, w)
        print(f'Add data point ({x:.5f}, {y:.5f})): \n')
        
        X = np.asarray([x ** j for j in range(n)]).reshape(-1, 1)
        y = np.array([[y]])
        S = np.linalg.pinv(var)
        
        estimate_var = np.linalg.pinv(a * (X @ X.T) + S)
        estimate_mu = estimate_var @ (a * X * y + S @ mu)
        print(f'Postirior mean:\n {estimate_mu}\nPosterior variance:\n {estimate_var}')

        pred_mu = (X.T@estimate_mu).item()
        pred_var = (X.T@estimate_var@X).item() + 1/a
        print(f'Predictive distribution ~ N({pred_mu:.5f}, {pred_var:.5f})')
        
        point_x.append(x)
        point_y.append(y.item())
        # pred_x.append(pred_mu)
        # pred_y.append(pred_var)
        if i in {10, 50, max_iter - 1}:
            visual_x.append(estimate_mu)
            visual_y.append(estimate_var)
        
        mu = estimate_mu
        var = estimate_var
        i += 1
    
    print(f'total sampling {i} point')
    return point_x, point_y, pred_x, pred_y, visual_x, visual_y, max_iter, n, a    


def plotResult(num_points, x_range, mu, var, title, a):
    pred_mu = np.zeros(len(x_range))
    pred_var = np.zeros(len(x_range))
    
    for i in range(len(x_range)):
        X = np.asarray([x_range[i] ** j for j in range(mu.shape[0])]).reshape(-1, 1)
        pred_mu[i] = (X.T @ mu).item()
        pred_var[i] = (X.T @ var @ X).item() + 1/a
        
    plt.scatter(point_x[:num_points], point_y[:num_points], color = 'blue')  # Data Point
    plt.plot(x_range, pred_mu, color = 'black')
    plt.plot(x_range, pred_mu - pred_var, color='red')
    plt.plot(x_range, pred_mu + pred_var, color='red')    
    
    plt.xlim(-2, 2)
    plt.ylim(-15, 20)
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(f'{title}')
    
    
point_x, point_y, pred_x, pred_y, visual_x, visual_y, max_iter, n, a = Baysian_Linear_Regression(b=1,n=4,a=1,w=[1,2,3,4])
plotResult(num_points=10, x_range=np.linspace(-2, 2, 500), mu=visual_x[0], var=visual_y[0], title='After 10 incomes', a=a)