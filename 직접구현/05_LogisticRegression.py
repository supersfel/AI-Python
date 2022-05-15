import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"   #한글
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
#---------------------------문제 1-----------------------------------
def Logistic(X):   #로지스틱 함수    
    return 1 / (1 + np.exp(-X)) 
x = np.linspace(-10,10,200) #로지스틱 함수 x입력값
Logistic_func = Logistic(x) #로지스틱 함수 생성

plt.plot(x, Logistic_func,label="Sigmoid") #그래프 그리기 
plt.grid(True, linestyle='--')
plt.legend()
plt.show()
#---------------------------문제 2-----------------------------------
raw_data = pd.read_csv('binary_data_insect.csv',encoding='utf-8', engine = 'python')
X = raw_data['Weight'].to_numpy()  #X데이터 가공
Xn = np.c_[X,np.ones(len(X))]  #행렬연산 가능하게 뒤에 1을 더해줌
Y = raw_data['Gender'].to_numpy() #Y데이터 가공
Yn = Y.reshape(len(X),1) #행렬연산이 가능하게 폼을 바꿈 
N = len(X) #데이터 개수 정의

plt.scatter(X[5:],Y[5:],marker= '^',label='수컷',facecolors='none', edgecolors='orange',s=80)
plt.scatter(X[:5],Y[:5],label='암컷',facecolors='none', edgecolors='b',s=80)
plt.grid(True, linestyle='--')
plt.legend()
plt.show() #암컷,수컷 그리기
#---------------------------문제 3-----------------------------------
def Cross_Entropy_Loss(Pn,Yn): #손실함수 정의
    return np.mean((Yn * np.log(Pn+1e-7) + (1-Yn)*np.log(1-Pn+1e-7))) * (-1)

def Gradient_Descent(lr,W,epoch): #경사하강법 함수정의  
    Ws,Cees,epochs = [[],[]],[],[] #그래프 생성을 위한 값 저장공간    
    for i in range(epoch):
        Pn = Logistic(W@Xn.T) #Pn값 갱신
        
        W0_diff = np.mean((Pn-Y) * X) #변화값 정의
        W1_diff = np.mean(Pn-Y)   
        
        W[0] = W[0] - lr * W0_diff #W0값 갱신
        W[1] = W[1] - lr * W1_diff #W1값 갱신
        
        cee = Cross_Entropy_Loss(Pn, Y) #cee구하기
        Ws[0].append(W[0])#값 저
        Ws[1].append(W[1])
        Cees.append(cee)
        epochs.append(i)
        if i%20000==0: #2만번마다 정보 출력       
            print('-----------------------')
            print('epoch:',i,'=====>','W :',W,', cee :',cee)
    print('GD 종료 learning rate :',lr) #종료 후 필요한 값들 딕셔너리로 반환
    return {'Ws' : Ws, 'CEE' : Cees, 'EPOCHS' : epochs, 'W':W}

result = Gradient_Descent(0.0005, [0,0], 250000)#경사하강법 실행

print('w0 =',result['W'][0],'w1 =',result['W'][1] )
plt.plot(result['EPOCHS'],result['CEE'])#CEE출력
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.show()
plt.plot(result['EPOCHS'],result['Ws'][0])#W0,W1 출력
plt.plot(result['EPOCHS'],result['Ws'][1])
plt.xlabel('Epoch')
plt.ylabel('CEE')
plt.show()
#---------------------------문제 4-----------------------------------
xn = np.linspace(30,90,200) #y예측값을 출력을 위해 생성
xn_ones = np.c_[xn,np.ones(len(x))]#행렬 연산을 위한 가공
Y_hat = result['W'] @ xn_ones.T #Decision boundary생성을 위함

def predict(INPUT,W): #predict함수 계싼
    INPUT = np.array([INPUT,1])
    Pn = Logistic(W@INPUT)
    if Pn > 0.5: Pn = 1
    else: Pn = 0
    return Pn
cnt=0 #정확도 예측을 위함
for i in range(N):
    if predict(X[i],result['W']) == Y[i]:
        cnt +=1
accuracy = cnt / N #케이스중 정답케이스를 확률로 계산
print('훈련결과 , 정확도 = ' , accuracy*100,"%",sep='')
for i in np.random.randint(low=0,high=90,size=20):#임의의 값 예측
    print(i,'값으로 예측 ====> :',end='')
    print('수컷입니다' if predict(i,result['W']) else '암컷입니다')
plt.plot(xn,Y_hat,label='Decision boundary') #z가 0이되는 지점(Decision boundary생성)
plt.scatter(X[5:],Y[5:],marker= '^',label='수컷',facecolors='none', edgecolors='orange',s=80)
plt.scatter(X[:5],Y[:5],label='암컷',facecolors='none', edgecolors='b',s=80)
plt.ylim([-0.25, 1.2])
plt.grid(True, linestyle='--')
plt.xlabel('Weight')
plt.ylabel('Gender')
plt.legend()
plt.show()
