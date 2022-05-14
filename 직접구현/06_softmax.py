import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
current_path = os.path.dirname(os.path.abspath(__file__))
raw_data = pd.read_csv('iris.csv',encoding='utf-8', engine = 'python')
X = raw_data[['sepal_length','petal_length']].to_numpy()    # x0 , x1 데이터 받기
N = len(X)
Xn = np.c_[X,np.ones(N)].T   # X데이터 있는 배열 뒤에 1 붙이기 ( 나중에 w구해서 계산하기 위함 )

Y = raw_data['variety'].to_numpy()
for i in range(N):
    if Y[i] == 'Versicolor':
        Y[i] = 1
    else:
        Y[i] = 0    
Yn = Y.reshape(N,1)

#---------------------------문제 1-----------------------------------
fig = plt.figure()
ax = fig.add_subplot(projection='3d')   # 3차원 그래프를 그리기 위함
ax.scatter(Xn[0],Xn[1],Yn)
ax.set_xlabel('sepal_length')
ax.set_ylabel('petal_length')
ax.set_zlabel('variety')
plt.show()   
#---------------------------문제 2-----------------------------------
def Cross_Entropy_Loss(Pn,Yn): #손실함수 정의
    return np.mean((Yn * np.log(Pn+1e-7) + (1-Yn)*np.log(1-Pn+1e-7))) * (-1)
def Logistic(X):   #로지스틱 함수    
    return 1 / (1 + np.exp(-X)) 
def Gradient_Descent(lr,W,epoch): #경사하강법 함수정의  
    Ws,Cees,epochs = [[],[],[]],[],[] #그래프 생성을 위한 값 저장공간    
    for i in range(epoch):
        Pn = Logistic(W@Xn) #Pn값 갱신
        
        W0_diff = np.mean((Pn-Y) * Xn[0]) #변화값 정의
        W1_diff = np.mean((Pn-Y) * Xn[1]) 
        W2_diff = np.mean(Pn-Y)
        
        W[0] = W[0] - lr * W0_diff #W0값 갱신
        W[1] = W[1] - lr * W1_diff #W1값 갱신
        W[2] = W[2] - lr * W2_diff
        
        cee = Cross_Entropy_Loss(Pn, Y) #cee구하기 
        Ws[0].append(W[0])#값 저
        Ws[1].append(W[1])
        Ws[2].append(W[2])
        Cees.append(cee)
        epochs.append(i)
        if i%1000==0: #2만번마다 정보 출력       
            print('-----------------------')
            print('epoch:',i,'=====>','W :',W,', cee :',cee) 
       
            
    print('GD 종료') #종료 후 필요한 값들 딕셔너리로 반환
    return {'Ws' : Ws, 'CEE' : Cees, 'EPOCHS' : epochs, 'W':W}

result = Gradient_Descent(0.003, [random.uniform(-3,3),random.uniform(-3,3),random.uniform(-3,3)], 20000)#경사하강법 실행

plt.plot(result['EPOCHS'],result['CEE'])#CEE출
plt.xlabel('Epoch')
plt.ylabel('CEE')
plt.show()

plt.plot(result['EPOCHS'],result['Ws'][0],label='W0')#W0,W1 출
plt.plot(result['EPOCHS'],result['Ws'][1],label='W1')
plt.plot(result['EPOCHS'],result['Ws'][2],label='W2')
plt.ylim([-4, 4])
plt.xlabel('Epoch')
plt.ylabel('weight')
plt.legend()
plt.show()

x1_data = np.linspace(4.5,7.0,1000)   # 55 ~ 190을 1000개로 나눈 배열
x2_data = np.linspace(1,5,1000)   # 10 ~ 100을 1000개로 나눈 배열
X1_data,X2_data = np.meshgrid(x1_data,x2_data)
Y_hat = result['W'][0] * X1_data + result['W'][1]* X2_data+result['W'][2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')   # 3차원 그래프를 그리기 위함
ax.plot_surface(X1_data,X2_data,Y_hat,cmap='plasma')
ax.scatter(Xn[0],Xn[1],Y)
ax.set_xlabel('sepal_length')
ax.set_ylabel('petal_length')
ax.set_zlabel('variety')
plt.show() 

Pn = Logistic(result['W'] @ Xn) #예측 Pn생성
cnt=0 #정확도 체크를 위한 cnt
for i in range(N): #문턱값보다 크면1,작으면0
    if Pn[i] > 0.5: Pn[i] = 1
    else: Pn[i] = 0
    if Pn[i] == Y[i]:
        cnt += 1 #정답과 일치하면 정답처리
accuracy = cnt / N #케이스중 정답케이스를 확률로 계
print('훈련결과 , 정확도 = ' , accuracy*100,"%",sep='')