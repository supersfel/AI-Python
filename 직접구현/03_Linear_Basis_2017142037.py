import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os

#실습5
raw_data = pd.read_csv('linear_regression_data01.csv', names=['age', 'tall'])
X = np.asarray(raw_data['age'].values.tolist())  # 나이 데이터(Xn)
Y = np.asarray(raw_data['tall'].values.tolist())  # 키 데이터(Yn)

N,Xmin,Xmax = len(X), min(X),max(X)

def Gaussian_linear(K):
    sigma = (Xmax - Xmin) / ( K - 1 )
    U = sigma * np.arange(K) + Xmin
    
    Gaussian_func=[]
    print(K,'sigma',sigma)
    
    for Uk in U:
        Gaussian_func.append( math.exp(1) ** (-0.5 * ( (X - Uk) / sigma )**2) )
    
    Gaussian_func = np.array(Gaussian_func)
    Gaussian_func = Gaussian_func.T
    print(Gaussian_func)
    Gaussian_func = np.c_[Gaussian_func,np.ones(N)]
    
   

    Gaussian_W = np.linalg.pinv(Gaussian_func.T @ Gaussian_func) @ Gaussian_func.T @ Y

    Gaussian_func_data = []
    X_data = np.linspace(Xmin, Xmax, 1000)
    
    for Uk in U:
        Gaussian_func_data.append( math.exp(1) ** (-0.5 * ( (X_data - Uk) / sigma )**2) )
    

    Gaussian_func_data = np.array(Gaussian_func_data)
    Gaussian_func_data = Gaussian_func_data.T
    Gaussian_func_data = np.c_[Gaussian_func_data, np.ones(1000)]

    MSE_Gaussian_Solution = np.mean((Gaussian_func @ Gaussian_W - Y) ** 2)
    MSEs.append(MSE_Gaussian_Solution)
    print('-------------------------------')
    print('K =',K )
    for idx in range(len(Gaussian_W)):
        print('w[',idx,'] =',Gaussian_W[idx],sep='')
    print('MSE_Gaussian_Solution = ' , MSE_Gaussian_Solution)

    #그래프 그리기
    Gaussian_Y = Gaussian_func_data @ Gaussian_W
    plt.grid(True, linestyle='--')
    plt.scatter(X, Y, color='green',label='original')  # 데이터 위치를 점으로 표시
    plt.scatter(X_data, Gaussian_Y,s=3,label='k= '+str(K) + ', MSE =' + str(MSE_Gaussian_Solution))  # 데이터 위치를 점으로 표시
    plt.title('Traing '+str(K))
    plt.xlabel('age')
    plt.ylabel('tall')
    plt.legend()


MSEs = [] # MSE 저장공간

plt.subplot(221)
Gaussian_linear(3)
plt.subplot(222)
Gaussian_linear(5)
plt.subplot(223)
Gaussian_linear(8)
plt.subplot(224)
Gaussian_linear(10)
plt.show()

plt.scatter([3,5,8,10], MSEs,label='MSE',color='dodgerblue')  # 데이터 위치를 점으로 표시

plt.plot([3,3],[0,MSEs[0]],color='dodgerblue')
plt.plot([5,5],[0,MSEs[1]],color='dodgerblue')
plt.plot([8,8],[0,MSEs[2]],color='dodgerblue')
plt.plot([10,10],[0,MSEs[3]],color='dodgerblue')
plt.grid(True, linestyle='--')
plt.xlim(2.5,10.5)
plt.ylim(0,0.6)
plt.title('MSE by K')
plt.xlabel('K')
plt.ylabel('MSE')
plt.show()