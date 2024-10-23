import math

def beta_binomial_conjugate(prior_a, prior_b, case):
    testfile = open('testcase.txt', 'r')
    line = testfile.readline().replace('\n', '')
    outputfile = open(f'beta_binomial_case{case}.txt', 'w')
    caseCount = 1

    while line:
        count_0 = line.count('0')
        count_1 = line.count('1')
        posterior_a = prior_a + count_1
        posterior_b = prior_b + count_0
        
        # binomial MLE
        p_mle = count_1 / (count_0 + count_1)  
        likelihood = math.factorial(count_0 + count_1) / (math.factorial(count_1) * math.factorial(count_0)) * p_mle**count_1 * (1-p_mle)**count_0
        
        # print(f'case {caseCount}: {line}')
        # print(f'Likelihood: {likelihood}')
        # print(f'Beta prior: a={prior_a} b={prior_b}')
        # print(f'Beta posterior: a={posterior_a} b={posterior_b}')
        outputfile.write(f'case {caseCount}: {line}\n')
        outputfile.write(f'Likelihood: {likelihood}\n')
        outputfile.write(f'Beta prior: a={prior_a} b={prior_b}\n')
        outputfile.write(f'Beta posterior: a={posterior_a} b={posterior_b}\n\n')

        prior_a = posterior_a
        prior_b = posterior_b
        caseCount += 1
        line = testfile.readline().replace('\n', '')
    
    testfile.close()
    outputfile.close()
    
# beta_binomial_conjugate(0, 0, 1)
beta_binomial_conjugate(10, 1, 2)


# def gamma_poisson_conjugate(prior_a, prior_b, case):
#     testfile = open('testcase.txt', 'r')
#     line = testfile.readline().replace('\n', '')
#     outputfile = open(f'gamma_poisson_case{case}.txt', 'w')
#     caseCount = 1
    
#     while line:
#         count_0 = line.count('0')
#         count_1 = line.count('1')
#         n = count_0 + count_1
        
#         posterior_a = prior_a + count_1
#         posterior_b = prior_b + n
#         theta_mle = count_1 / n
#         likelihood = math.exp(- n * theta_mle) * theta_mle ** count_1 / math.prod([math.factorial(int(yi)) for yi in line])
        
#         # print(f'case {caseCount}: {line}')
#         # print(f'Likelihood: {likelihood}')
#         # print(f'Beta prior: a={prior_a} b={prior_b}')
#         # print(f'Beta posterior: a={posterior_a} b={posterior_b}')
        
#         outputfile.write(f'case {caseCount}: {line}\n')
#         outputfile.write(f'Likelihood: {likelihood}\n')
#         outputfile.write(f'Beta prior: a={prior_a} b={prior_b}\n')
#         outputfile.write(f'Beta posterior: a={posterior_a} b={posterior_b}\n\n')

#         prior_a = posterior_a
#         prior_b = posterior_b
#         caseCount += 1
#         line = testfile.readline().replace('\n', '')
    
#     testfile.close()
#     outputfile.close()
    
# gamma_poisson_conjugate(0, 0, 1)
# gamma_poisson_conjugate(10, 1, 2)