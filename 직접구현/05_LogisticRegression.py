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
            

x = np.linspace(-10,10,200)
Logistic_func = Logistic(x)


plt.plot(x, Logistic_func,label="Sigmoid")  
plt.grid(True, linestyle='--')
plt.legend()
plt.show()
#---------------------------문제 2-----------------------------------

raw_data = pd.read_csv('binary_data_insect.csv',encoding='utf-8', engine = 'python')

X = raw_data['Weight'].to_numpy()  
Xn = np.c_[X,np.ones(len(X))]  
Y = raw_data['Gender'].to_numpy()
Yn = Y.reshape(len(X),1)
plt.scatter(X[5:],Y[5:],marker= '^',label='수컷',facecolors='none', edgecolors='orange',s=80)
plt.scatter(X[:5],Y[:5],label='암컷',facecolors='none', edgecolors='b',s=80)
plt.grid(True, linestyle='--')
plt.legend()
plt.show()

#---------------------------문제 3-----------------------------------

def Cross_Entropy_Loss(Pn,Yn):
    return np.mean((Yn * np.log(Pn+1e-7) + (1-Yn)*np.log(1-Pn+1e-7))) * (-1)

def Gradient_Descent(lr,W,epoch):
    Pn = Logistic(W @ Xn.T)        
    
    print('CEE' , Cross_Entropy_Loss(Pn, Y))
    Ws = [[],[]]
    Cees = []
    epochs = []
    
    for i in range(epoch):
        Pn = Logistic(W@Xn.T)
        
        W0_diff = np.mean((Pn-Y) * X)
        W1_diff = np.mean(Pn-Y)   
        
        W[0] = W[0] - lr * W0_diff
        W[1] = W[1] - lr * W1_diff    
        
        cee = Cross_Entropy_Loss(Pn, Y)
        Ws[0].append(W[0])
        Ws[1].append(W[1])
        Cees.append(cee)
        epochs.append(i)
        if i%100000==0:
            
            print('-----------------------')
            print('epoch:',i,'=====>','W :',W,', cee :',cee)
            
            
    print('finish')
    return {'W' : Ws, 'CEE' : Cees, 'EPOCHS' : epochs}

#Gradient_Descent(0.00000055, [0.11723166996504109,-5.6480077099576045], 1000000)
print('X',X)
print('Xn',Xn)
print('Y',Y)
print('Yn',Yn)
Pn = Logistic(Xn@[0.11723166996504109,-5.6480077099576045])
print('Pn',Pn)

result = Gradient_Descent(0.000055, [0,0], 5000000)




plt.plot(result['EPOCHS'],result['CEE'])
plt.plot(result['EPOCHS'],result['W'][0])
plt.plot(result['EPOCHS'],result['W'][1])

