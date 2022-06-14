import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
current_path = os.path.dirname(os.path.abspath(__file__))
raw_data = pd.read_csv('multiple_linear_regression_data.csv',encoding='utf-8', engine = 'python')
X = raw_data[['height','weight']].to_numpy()    # x0 , x1 데이터 받기
X = np.c_[X,np.ones(80)]   # X데이터 있는 배열 뒤에 1 붙이기 ( 나중에 w구해서 계산하기 위함 )
Y = raw_data['label'].to_numpy()
Y = Y.reshape(80,1)   # 행렬계산을 위해 맞춰줌

#------------------------------------------------------------------
#                          실습1
#------------------------------------------------------------------
height = raw_data['height'].to_numpy()  # X0데이터
weight = raw_data['weight'].to_numpy()  # X1 데이터
age = raw_data['label'].to_numpy()    #Y 데이터
N = len(height)   # N = 80 ( 데이터 수 )

fig = plt.figure()
ax = fig.add_subplot(projection='3d')   # 3차원 그래프를 그리기 위함
ax.scatter(height,weight,age)
ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('age')
#plt.show()   # 마지막에 한번에 출력하기위해 주석
#------------------------------------------------------------------
#                           실습2
#------------------------------------------------------------------
analytic_W = np.linalg.pinv(X.T @ X) @ X.T @ Y    # W=((pi)T*(pi))^-1(pi)T*y
height_data = np.linspace(55,190,1000)   # 55 ~ 190을 1000개로 나눈 배열
weight_data = np.linspace(10,100,1000)   # 10 ~ 100을 1000개로 나눈 배열
Height,Weight = np.meshgrid(height_data,weight_data)
analytic_y = analytic_W[0]*Height + analytic_W[1]*Weight + analytic_W[2]   # W0 * X0 + W1 * X1 + W2 식 구현
print(analytic_W)

fig = plt.figure()   # 그래프 그리기
ax = fig.add_subplot(projection='3d')
ax.scatter(height,weight,age)
ax.plot_surface(Height,Weight,analytic_y,cmap='plasma')
ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('age')
ax.set_title('Analytic Solution')   #해석해로 구한 평면
#plt.show()  # 마지막에 한번에 출력하기위해 주석

#------------------------------------------------------------------
#                           실습3
#------------------------------------------------------------------
MSE_Analytic_Solution = np.mean( (X @ analytic_W - Y) ** 2 )  # 해석해로 구한 식의 MSE값을 행렬로 계산
print('----------------')
print('실습3')
print('MSE_Analytic_Solution :',MSE_Analytic_Solution)
print('----------------')

#------------------------------------------------------------------
#                           실습4
#------------------------------------------------------------------
print('실습4')
def Gradient_Descent(lr,Gradient_W,epoch):    #경사하강법 함수 learning rate , epoch 와 W들을 인자로 받음
    MSE_Gradient_Descent = np.mean((X @ Gradient_W - age) ** 2)      #경사하강법으로 구한 해석해의 MSE
    print('before MSE_Gradient_Descent :', MSE_Gradient_Descent)     #학습 시키기 전의 MSE
    print('')

    for i in range(epoch):   # epoch만큼 반복
        y_pred = X @ Gradient_W   # 바뀐 W들로 구한 예측값
        error = y_pred - age      # 예측값과의 차이

        W0_diff = np.mean((height * error))    # 각 W들의 기울기 구하기
        W1_diff = np.mean((weight * error))
        W2_diff = np.mean(error)

        Gradient_W[0] = Gradient_W[0] - lr * W0_diff    # W들 값 갱신
        Gradient_W[1] = Gradient_W[1] - lr * W1_diff
        Gradient_W[2] = Gradient_W[2] - lr * W2_diff

        if i%500 ==0:   # 500번 학습할때마다 W0,W1,W2값과 MSE값 출력
            MSE_Gradient_Descent = np.mean((X @ Gradient_W - age) ** 2)  # 학습을 시켰으므로 MSE값 갱신
            print( "epoch",i, '====> W0 =',Gradient_W[0],'W1 =',Gradient_W[1],'W2 =',Gradient_W[2],'MSE =',MSE_Gradient_Descent)

    print('')
    print('END MSE_Gradient_Descent :', MSE_Gradient_Descent)   #학습 종료시 MSE출력
    return Gradient_W

Gradient_W = Gradient_Descent(lr=0.000055,Gradient_W=[100,100,0],epoch=5000000)   #넣어준 값들로 학습시킨 후 Gradient_W에 저장

print('----------------')
print('W0 =',Gradient_W[0],'W1 =',Gradient_W[1],'W2 =',Gradient_W[2])  #학습 완료 후 W값들 출력
print('----------------')

gradient_y = Gradient_W[0]*Height + Gradient_W[1]*Weight + Gradient_W[2]  #경사하강법으로 구한 W를 사용하여 구한 모델
print(Gradient_W)
print('------------')
print(Height)
fig = plt.figure()   #모델 출력
ax = fig.add_subplot(projection='3d')
ax.scatter(height,weight,age)
ax.plot_surface(Height,Weight,gradient_y,cmap='plasma')
ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('age')
ax.set_title('Gradient Desent')
plt.show()

