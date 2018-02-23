import theano
import theano.tensor as T
#theano.config.compute_test_value = 'off'

import numpy as np


# routine to make some data up
def makeup_data(n,p):
    X = np.memmap('test.memmap', dtype='float64', mode='w+', shape=(n,p))
    X[:,:] = np.random.normal(size=n*p).reshape(n,p)
    X[:,:] = (X - np.mean(X,0))/np.std(X,0)
    b = np.random.normal(size=p)
    b[:] = 0
    b[3:5] =  10.0
    y = 2.5 + np.dot(X,b) + np.random.normal(0,0.1,size=n)
    return y, b, X

##### define the symbolic variables

X = T.dmatrix('X')
gamma = T.vector('gamma')
beta = T.vector('beta')
y = T.vector('y')
ebv = T.vector('ebv')
b0 = T.scalar('b0')
v0 = T.scalar('v0')
v = T.vector('v')

#######  Try: update beta and EBV
def oneStep(X_j_col, gamma_j, beta_j, vj, pre_ebv, y, b0, v0): #(X_j_col,gamma_j, beta_j, vj,y,ebv,b0,v0):
    T.printing.Print('X_j_col')(X_j_col)
    fromEBV = X_j_col * gamma_j * beta_j
    new_b_j = T.sum((gamma_j * X_j_col) * (y - pre_ebv - b0 + fromEBV) ) / (T.sum((gamma_j * X_j_col)**2) + (v0/vj))
    toEBV = X_j_col * gamma_j * new_b_j
    new_ebv  = pre_ebv - fromEBV + toEBV
    return new_b_j, new_ebv

[new_beta,new_EBV], u2 = theano.scan(fn=oneStep ,
                                     outputs_info=[None, ebv],
                                     sequences = [X,gamma, beta,v],
                                     non_sequences=[y,b0,v0])

final_ebv = new_EBV[-1]
calculate_beta_and_EBV = theano.function(inputs=[X,gamma, beta,v,ebv,y,b0,v0], outputs=[new_beta,final_ebv],updates=u2)


# generate some data. just to run the code
# define number of rows and number of columns
n = 9
p = 10

print('Making data up')
y,b,X = makeup_data(n=n,p=p)

# initialise the variables
beta= np.random.normal(0,0.001,size=p)
v = np.ones(p)*0.1
lambda_ =1.0
v_0= np.var(y)
beta_0 = np.mean(y)/2.0
gamma = np.ones(p)*0.5
EBV = np.sum(X*gamma*beta,1)

print('THEANO RESULTS')

print('update beta + EBV')
b2,ebv2 = calculate_beta_and_EBV(X.T,gamma, beta,v,EBV,y,beta_0,v_0)#(X.T,gamma, beta,y,EBV,beta_0,v_0,v)
print('beta',b2)
print('EBV',ebv2)

print('PYTHON RESULTS')
print('Loop update beta + EBV')
def run_loop(X,gamma, beta,y,EBV,beta_0,v_0,v):
    for j in range(int(p)):
        fromEBV = X[:,j] * gamma[j] * beta[j]
        beta[j] = np.sum((gamma[j] * X[:,j]) * (y - EBV - beta_0 + fromEBV) ) / (np.sum((gamma[j] * X[:,j])**2) + (v_0/v[j]))
        toEBV = X[:,j] * (gamma[j] * beta[j])
        EBV = EBV - fromEBV + toEBV
    return beta, EBV
b3, ebv3 = run_loop(X,gamma, beta,y,EBV,beta_0,v_0,v)
print('beta', b3)
print('EBV',ebv3)


print('TIMING')
print('PYTHON VERSION')

#%timeit run_loop(X,gamma, beta,y,EBV,beta_0,v_0,v)

print('THEANO VERSION')

#%timeit calculate_beta_and_EBV(X.T,gamma, beta,v,EBV,y,beta_0,v_0)