import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#---------------------------실습 1-----------------------------------
raw_data = pd.read_csv('lin_regression_data_03.csv', names=['age', 'tall'])
X = np.asarray(raw_data['age'].values.tolist())  # 나이 데이터(X)
Y = np.asarray(raw_data['tall'].values.tolist())  # 키 데이터(Y)
min_Xdata,max_Xdata = min(X),max(X)

plt.scatter(X, Y,label="infant's age and height data")  # 데이터 위치를 점으로 표시
plt.xlabel('age[month]')
plt.ylabel('height[cm]')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

#---------------------------실습 2-----------------------------------

Training_X,Training_Y= X[:20],Y[:20] #트레이닝 셋( 1~20 )
Test_X,Test_Y = X[20:],Y[20:] #테스트 셋 ( 21 ~ 25 )

plt.scatter(Training_X, Training_Y,label="Training data")  # 데이터 위치를 점으로 표시
plt.scatter(Test_X, Test_Y,label="Test data")  # 데이터 위치를 점으로 표시
plt.xlabel('age[month]')
plt.ylabel('height[cm]')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

#---------------------------실습 3 , 4-----------------------------------

def Gaussian_linear(K,X,Y,test_x,test_y):  # 가우스 함수를 이용한 선형 기저함수 회귀모델를 보기위한 함수 K개의 가우스 함수를 사용할지 인자로 받음
    N,Xmax,Xmin = len(X),max(X),min(X)
    sigma = (Xmax - Xmin) / (K - 1)  # 시그마 값 계산
    U = sigma * np.arange(K) + Xmin  # k가 0~K-1까지 들어있는 Uk를 모음

    Gaussian_func = []  # 가우스 함수들을 저장할 배열 생성
    for Uk in U:
        Gaussian_func.append(np.exp(-0.5 * ((X - Uk) / sigma) ** 2))  # 가우스 함수들을 저장

    Gaussian_func = np.array(Gaussian_func)
    Gaussian_func = Gaussian_func.T  # 저장된 값들이 행이아닌 열기준으로 함수가 저장되기에 전치행렬을 해줌
    Gaussian_func = np.c_[Gaussian_func, np.ones(N)]  # 예측값 계산을 위해 뒤에 1인 배열을 붙여줌
    Gaussian_W = np.linalg.pinv(Gaussian_func.T @ Gaussian_func) @ Gaussian_func.T @ Y  # 식에 의해서 구한 W값 계산
    MSE_Gaussian_Solution = np.mean((Gaussian_func @ Gaussian_W - Y) ** 2)  # 가우스 함수를 이용한 선형 기저함수 모델의 MSE계산
    Training_MSEs.append(MSE_Gaussian_Solution)

    test_data = []  # 테스트 집합에 적용
    for Uk in U:
        test_data.append(np.exp(-0.5 * ((test_x - Uk) / sigma) ** 2))  # 위와 같은작업 반복

    test_data = np.array(test_data)  # 위와 같은작업 반복
    test_data = test_data.T  # 저장된 값들이 행이아닌 열기준으로 함수가 저장되기에 전치행렬을 해줌
    test_data = np.c_[test_data, np.ones(len(test_x))]  # 예측값 계산을 위해 뒤에 1인 배열을 붙여줌
    Test_Set_MSE_Gaussian_Solution = np.mean((test_data @ Gaussian_W - test_y) ** 2)  # 테스트집합의 MSE계산
    Test_MSEs.append(Test_Set_MSE_Gaussian_Solution)

    Gaussian_func_data = []  # 예측값들을 무작위로 보기위한 데이터들
    X_data = np.linspace(min_Xdata, max_Xdata, 1000)
    for Uk in U:
        Gaussian_func_data.append(np.exp(-0.5 * ((X_data - Uk) / sigma) ** 2))  # 위와 같은작업 반복

    Gaussian_func_data = np.array(Gaussian_func_data)  # 위와 같은작업 반복
    Gaussian_func_data = Gaussian_func_data.T  # 위와 같은작업 반복
    Gaussian_func_data = np.c_[Gaussian_func_data, np.ones(1000)]  # 위와 같은작업 반복
    Gaussian_func_Model = Gaussian_func_data @ Gaussian_W

    print('-------------------------------')  # 구한 W들을 차례대로 출력
    print('K =', K)
    print(Gaussian_W)
    print('MSE_Gaussian_Solution = ', MSE_Gaussian_Solution)
    print('Test_Set_MSE_Gaussian_Solution = ', Test_Set_MSE_Gaussian_Solution)
    return { 'Gaussian_W' : Gaussian_W ,
             'Gaussian_func_Model' : Gaussian_func_Model ,
             'X_data' : X_data,
             'Test_Set_MSE' : Test_Set_MSE_Gaussian_Solution
        }

Training_MSEs,Test_MSEs=[],[]
for i in range(6,14):
    Gaussian_linear(i,Training_X,Training_Y,Test_X,Test_Y)
plt.plot(range(6,14),Training_MSEs,label='training MSE')
plt.plot(range(6,14),Test_MSEs,label='Test MSE')
plt.xlabel('K')
plt.ylabel('MSE')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

#---------------------------실습 5-----------------------------------
Set_X = np.array_split(X,5)  #5개로 각 데이터를 나눔
Set_Y = np.array_split(Y,5)

for i in range(5):
    plt.scatter(Set_X[i],Set_Y[i],label=str(i)+'th set')  # 데이터 위치를 점으로 표시
plt.xlabel('age[month]')
plt.ylabel('height[cm]')
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

#---------------------------실습 6,7-----------------------------------
plot_order=231
for i in range(5):
    print('-----------------------------------------------------------------')
    print('-----------------------------------------------------------------')
    HoldOut_Test_X = Set_X[i]
    HoldOut_Test_Y = Set_Y[i]
    HoldOut_Training_X = np.array([])
    HoldOut_Training_Y = np.array([])
    for  j in range(5):
        if i != j:
            HoldOut_Training_X = np.concatenate([HoldOut_Training_X, Set_X[j]])
            HoldOut_Training_Y = np.concatenate([HoldOut_Training_Y, Set_Y[j]])

    Model = Gaussian_linear(9,HoldOut_Training_X,HoldOut_Training_Y,HoldOut_Test_X,HoldOut_Test_Y)
    plt.subplot(plot_order)
    plot_order+=1
    plt.grid(True, linestyle='--')
    plt.scatter(HoldOut_Test_X, HoldOut_Test_Y, color='orange', label='validation set')  # 데이터 위치를 점으로 표시
    plt.scatter(HoldOut_Training_X, HoldOut_Training_Y, color='green', label='training set')  # 데이터 위치를 점으로 표시
   
    plt.scatter(Model['X_data'], Model['Gaussian_func_Model'], s=3,label='k='+str(i)+',MSE='+str(Model['Test_Set_MSE']))  # 라벨에 K와 MSE를 표시
    plt.xlabel('age')
    plt.ylabel('tall')
    plt.legend()
plt.show()

