import math
import random

import numpy as np
# Assume that K=2
if __name__ == '__main__':
    #mean = (1, 2, 0, 3, 0, 4, 5, 6)

    # random_list=[]
    # random_list1 = []
    # num_features=8
    # upper=3
    # for i in range(num_features):
    #     r=random.uniform(1,upper)
    #     while r in random_list:
    #         r=random.uniform(1,upper)
    #     random_list.append(r)
    # for i in range(num_features):
    #     r=random.uniform(1,upper)
    #     while r in random_list1:
    #         r=random.uniform(1,upper)
    #     random_list1.append(r)
    # num_irrelevant=2
    # for i in range(num_irrelevant):
    #     random_list[i]=0
    #     random_list1[i]=0
    # print(random_list)
    # print(random_list1)
    # random_list=[0, 0, 1, 2, 3, 4, 5,
    #  6]
    # random_list1=[0, 0, 2, 4, 1, 3, 6,
    #  5]
    mean=[0, 0, 1, 2, 3, 4, 5, 6]
    #mean = [1,2]
    theta=1
    num=500
    cov = []
    for i in range(len(mean)):
        temp=[]
        for j in range(len(mean)):
            if i==j:
                temp.append(math.pow(theta,2))
            else:
                temp.append(0)
        cov.append(temp)
    x = np.random.multivariate_normal(mean, cov, (num), 'raise')
    x=np.insert(x,len(mean),values=0,axis=1)
    # print("negative",x)
    #mean1 = (2, 3, 0, 2.5, 0, 3.7,5.2, 5.8)
    mean1=[0, 0, 2, 4, 1, 3, 6, 5]
    #mean1 = [2,3]
    x1 = np.random.multivariate_normal(mean1, cov, (num), 'raise')
    x1=np.insert(x1,len(mean),values=1,axis=1)
    # print("positive",x1)
    y=np.r_[x,x1]
    np.savetxt('test.csv', y, delimiter = ',')
    print("Data set simulation complete...")