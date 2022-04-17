import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#------------------------------------------------------------------
#                        실습5,6
#------------------------------------------------------------------
raw_data = pd.read_csv('linear_regression_data01.csv', names=['age', 'tall'])
X = np.asarray(raw_data['age'].values.tolist())  # 나이 데이터(Xn)
Y = np.asarray(raw_data['tall'].values.tolist())  # 키 데이터(Yn)

N,Xmin,Xmax = len(X), min(X),max(X)   #계산시 반복되는 값들을 미리 저장

def Gaussian_linear(K):    #가우스 함수를 이용한 선형 기저함수 회귀모델를 보기위한 함수 K개의 가우스 함수를 사용할지 인자로 받음
    sigma = (Xmax - Xmin) / ( K - 1 )   #시그마 값 계산
    U = sigma * np.arange(K) + Xmin     #k가 0~K-1까지 들어있는 Uk를 모음
    
    Gaussian_func=[] #가우스 함수들을 저장할 배열 생성
    for Uk in U:
        Gaussian_func.append( np.exp(-0.5 * ( (X - Uk) / sigma )**2) )   #가우스 함수들을 저장
    
    Gaussian_func = np.array(Gaussian_func)
    Gaussian_func = Gaussian_func.T  # 저장된 값들이 행이아닌 열기준으로 함수가 저장되기에 전치행렬을 해줌
    Gaussian_func = np.c_[Gaussian_func,np.ones(N)] #예측값 계산을 위해 뒤에 1인 배열을 붙여줌

    Gaussian_W = np.linalg.pinv(Gaussian_func.T @ Gaussian_func) @ Gaussian_func.T @ Y # 식에 의해서 구한 W값 계산

    Gaussian_func_data = []   # 예측값들을 무작위로 보기위한 데이터들
    X_data = np.linspace(Xmin, Xmax, 1000)
    for Uk in U:
        Gaussian_func_data.append( np.exp(-0.5 * ( (X_data - Uk) / sigma )**2) )  #위와 같은작업 반복

    Gaussian_func_data = np.array(Gaussian_func_data)  #위와 같은작업 반복
    Gaussian_func_data = Gaussian_func_data.T  #위와 같은작업 반복
    Gaussian_func_data = np.c_[Gaussian_func_data, np.ones(1000)]  #위와 같은작업 반복

    MSE_Gaussian_Solution = np.mean((Gaussian_func @ Gaussian_W - Y) ** 2)   #가우스 함수를 이용한 선형 기저함수 모델의 MSE계산
    MSEs.append(MSE_Gaussian_Solution) #구한 MSE들을 저장
    print('-------------------------------')    #구한 W들을 차례대로 출력
    print('K =',K )
    for idx in range(len(Gaussian_W)):
        print('w[',idx,'] =',Gaussian_W[idx],sep='')
    print('MSE_Gaussian_Solution = ' , MSE_Gaussian_Solution)

    #그래프 그리기
    Gaussian_Y = Gaussian_func_data @ Gaussian_W
    plt.grid(True, linestyle='--')
    plt.scatter(X, Y, color='green',label='original')  # 데이터 위치를 점으로 표시
    plt.scatter(X_data, Gaussian_Y,s=3,label='k= '+str(K) + ', MSE =' + str(MSE_Gaussian_Solution))  # 라벨에 K와 MSE를 표시
    plt.title('Traing '+str(K))
    plt.xlabel('age')
    plt.ylabel('tall')
    plt.legend()

MSEs = [] # MSE 저장공간
plt.subplot(221)  #한 화면 안에 그래프 4개를 그리기 위함
Gaussian_linear(3)   #k=3 일때 가우스 함수를 이용한 선형 기저함수 모델
plt.subplot(222)
Gaussian_linear(5)   #k=5 일때 가우스 함수를 이용한 선형 기저함수 모델
plt.subplot(223)
Gaussian_linear(8)   #k=8 일때 가우스 함수를 이용한 선형 기저함수 모델
plt.subplot(224)
Gaussian_linear(10)  #k=10 일때 가우스 함수를 이용한 선형 기저함수 모델
plt.show()
#------------------------------------------------------------------
#                         실습 7
#------------------------------------------------------------------
# 강의자료와 유사하게 출력하기 위해 점을 찍고 점에서 수직으로 그래프를 그림
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