import numpy as np
import math as m
K=3                                                                             #K = Number of gaussians
M=1000                                                                          #M = Number of training

theta=[[0 for x in range (K)] for y in range (3)]                               #prob,mean(x,y),cov
theta[0][0:] = 0.3, (0,1), [[1,0], [0,1]]
theta[1][0:] = 0.5, (100,0), [[2,0], [0,2]]
theta[2][0:] = (1-theta[0][0]-theta[1][0]), (3,3), [[3,0],[0,3]]                #"theta fill in with values, for large K can replace with for loop and random"

Z=np.random.choice(K, M, p=[theta[0][0], theta[1][0], theta[2][0]])+1           #"generates 'M' size random vector with values 1:'K' with the probabilities "

x_y=np.zeros([M,2])                                                             #Create list with M lines, each is ndarray size of 2 of x,y.

for i in range(M):
    mean = theta[Z[i]-1][1]
    cov = theta[Z[i]-1][2]
    x_y[i]=(np.random.multivariate_normal(mean,cov,1))
    w

nd = lambda x: 1/m.sqrt((2*pi)^2*np.linalg.det())*m.exp(-((x-mu[x])^2)/(2*sigma[x]))
