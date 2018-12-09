import numpy as np
from numpy.linalg import inv
import math as m
K=3                                                                            #K = Number of gaussians
M=900                                                                          #M = Number of training-data

theta_real=[[0 for x in range (K)] for y in range (3)]                         #from theta real we generate the 2D data points
theta_real[0][0:] = 0.3, (0,0), [[0.5,0], [0,0.5]]
theta_real[1][0:] = 0.3, (2,-2), [[0.5,0], [0,0.5]]
theta_real[2][0:] = (1-theta_real[0][0]-theta_real[1][0]), (-3,3), [[0.5,0],[0,0.5]]

theta=[[0 for x in range (K)] for y in range (3)]                              #prob,mean(x,y),cov
theta[0][0:] = 0.35, (0.5,0.5), [[1,0], [0,1]]
theta[1][0:] = 0.25, (2.7,-2.7), [[0.8,0], [0,0.8]]
theta[2][0:] = (1-theta[0][0]-theta[1][0]), (-3.5,2.5), [[1,0],[0,1]]          #"theta fill in with values, for large K can replace with for loop and random"

Z=np.random.choice(K, M, p=[theta_real[0][0], theta_real[1][0], theta_real[2][0]])+1           #"generates 'M' size random vector with values 1:'K' with the probabilities "
x_y=np.zeros([M,2])                                                            #Create list with M lines, each is ndarray size of 2 of x,y.
w=np.zeros([M,K])                                                              #Create zero vector for w
label_GMM=np.zeros([1,M])
label_Kmeans=np.zeros([1,M])

def rand_2D(M,Z,theta,x_y):
    for i in range(M):                                                              #Generate M 2D points and calculate w for each.
        mean = theta_real[Z[i]-1][1]
        cov = theta_real[Z[i]-1][2]
        x_y[i]=(np.random.multivariate_normal(mean,cov,1))
    return x_y
def Expectation_step(K,M,theta,x_y,w): #E-Step function
    for i in range (M):
        expo = [None] * K
        w_i_denumerator = 0
        for j in range(K):
            xy_minus_means=(np.asarray(x_y[i]-theta[j][1]))
            expo[j]=np.matmul(np.matmul(xy_minus_means.T,inv(theta[j][2])),xy_minus_means)
            w_i_denumerator+=theta[j][0] * (1 / m.sqrt((2 * m.pi) ** 2 * np.linalg.det(theta[j][2])) * m.exp(-0.5 * expo[j]))
        for j in range(K):
            w_i_numerator = theta[j][0] * (1 / m.sqrt(((2 * m.pi) ** 2) * np.linalg.det(theta[j][2]))) * m.exp(-0.5 * expo[j])
            if w_i_numerator == 0:
                  print('Something is wrong: Wi = 0')
            w[i][j] = (w_i_numerator / w_i_denumerator)
def Maximization_step(K,M,w,theta):
    for j in range (K):
        sum = sum2 = sum3 = 0
        for m in range (M):
            sum+=w[m][j]
            sum2+=w[m][j]*x_y[m]
            sum3+=w[m][j]*np.matmul((theta[j][1]-x_y[m].reshape(-1,2)).T,(theta[j][1]-x_y[m].reshape(-1,2)))
        theta[j][0]=sum/M
        theta[j][1]=sum2/sum
        theta[j][2]=sum3/sum
    return
def K_Means(K,M,theta,x_y,num_iter):
    m = np.zeros([2,K])
    for ind in range(K): #initialize m
        m[0][ind] = theta[ind][1][0]
        m[1][ind] = theta[ind][1][1]
    for itr in range(num_iter):
        K_dist = np.zeros([M, K])
        cnt = np.zeros([1, K])
        sum = np.zeros([2, K])
        for i in range (M):
            for j in range(K): #fill matrix 'K' of distances
                K_dist[i][j]=np.sqrt((x_y[i][0]-m[0][j])**2)+np.sqrt((x_y[i][1]-m[1][j])**2)
            label_Kmeans[0][i]=np.argmin(K_dist[i])+1 #fill labels with the closer distance
            for j in range(K): #Prepare the values to update m
                if label_Kmeans[0][i]==j+1:
                    cnt[0][j]+=1
                    sum[0][j]+=x_y[i][0]
                    sum[1][j]+=x_y[i][1]
        for j in range(K):
            m[0][j]=sum[0][j]/cnt[0][j]
            m[1][j]=sum[1][j]/cnt[0][j]
    return
def GMM(K,M,theta,x_y,w,num_iter):
    for i in range(num_iter):
        Expectation_step(K, M, theta, x_y, w)
        Maximization_step(K, M, w, theta)
    for i in range(M):
        label_GMM[0][i] = int(np.argmax(w[i]) + 1)
    return
def Success_Rate(label,Z,iter):
    sccs=(len(Z)-np.count_nonzero(label-Z))/len(Z)*100
    print('The success rate of GMM at the')
x_y = rand_2D(M,Z,theta,x_y)
num_iter=10
GMM(K,M,Z,theta,x_y,w,num_iter)
K_Means(K,M,theta,x_y,num_iter)

#print(sum(np.bitwise_xor(pred_label.reshape(-1,1),Z.T)))
print('x')