from RandomDataGenerator import Random_data_Generator

def Sequential_Estimator(m, s):
    print(f'Data point source function: N({m}, {s})')
    n = 0
    estimate_mu, estimate_var = 0, 0
    threshold = 1e-3
    
    while (abs(m-estimate_mu) > threshold or abs(s-estimate_var) > threshold) and n < 10000:
        update_x = Random_data_Generator(m, s)
        n += 1
        
        update_mu = estimate_mu + (update_x - estimate_mu) / n
        update_var = (1 - 1/n) * estimate_var + (update_x - estimate_mu) * (update_x - update_mu) / n
        
        estimate_mu = update_mu
        estimate_var = update_var
        
        print(f'Add datapoint: {update_x}\nMean = {estimate_mu} Var = {estimate_var}')
    print(f'total sampling {n} point')
        
Sequential_Estimator(3, 5)

