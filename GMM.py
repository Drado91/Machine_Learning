import numpy as np
from numpy.linalg import inv
import math as m
K=3                                                                             #K = Number of gaussians
M=1000                                                                          #M = Number of training

theta=[[0 for x in range (K)] for y in range (3)]                               #prob,mean(x,y),cov
theta[0][0:] = 0.3, (0,1), [[1,0], [0,1]]
theta[1][0:] = 0.5, (100,0), [[2,0], [0,2]]
theta[2][0:] = (1-theta[0][0]-theta[1][0]), (3,3), [[3,0],[0,3]]                #"theta fill in with values, for large K can replace with for loop and random"

Z=np.random.choice(K, M, p=[theta[0][0], theta[1][0], theta[2][0]])+1           #"generates 'M' size random vector with values 1:'K' with the probabilities "
x_y=np.zeros([M,2])                                                             #Create list with M lines, each is ndarray size of 2 of x,y.

def w_calc(i):
    expo = [None] * K
    for j in range(K):
        xy_minus_means=(np.asarray(x_y[i]-theta[Z[j]-1][1]))0
        expo[j]=np.matmul(np.matmul(xy_minus_means.T,inv(theta[Z[j]-1][2])),xy_minus_means)
       # w_i_numerator = theta[Z[i] - 1][0] * 1 / m.sqrt((2 * m.pi) ** 2 * np.linalg.det(theta[Z[i] - 1][2])) * m.exp(-0.5 * expo)
        #w_i_denumerator = theta[Z[0] - 1][0] * 1 / m.sqrt((2 * m.pi) ** 2 * np.linalg.det(theta[Z[0] - 1][2])) * m.exp(-0.5 * expo)
    return w_i

for i in range(M):
    mean = theta[Z[i]-1][1]
    cov = theta[Z[i]-1][2]
    x_y[i]=(np.random.multivariate_normal(mean,cov,1))
    w[i]=w_calc(i)




#ND = lambda i :
#print(ND(100))
print(3)