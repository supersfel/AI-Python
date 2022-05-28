import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class perceptron_for_GATE:   #GATE만들기위한 퍼셉트론
    def __init__(self,w):
        self.w = w
    def output(self,x):
        tmp = np.dot(self.w,np.append(1,x))
        result = 1.0*(tmp>0)
        return result
#======================실습1========================
w_and = np.array([-1.2,1,1])   #AND GATE
and_gate = perceptron_for_GATE(w_and)
w_or = np.array([-0.8,1,1]) # OR GATE
or_gate = perceptron_for_GATE(w_or)
w_nand = np.array([1.2,-1,-1]) #NAND GATE
nand_gate = perceptron_for_GATE(w_nand)

x_lst=[[0,0],[1,0],[0,1],[1,1]] #입력
print('========AND GATE========')
for x in x_lst:
    print(f'x = {x} ====> {and_gate.output(x)}')
print('========OR GATE========')
for x in x_lst:
    print(f'x = {x} ====> {or_gate.output(x)}')
print('========NAND GATE========')
for x in x_lst:
    print(f'x = {x} ====> {nand_gate.output(x)}')
print('========XOR GATE========')
for x in x_lst:  #XOR GATE
    s1 = nand_gate.output(x)
    s2 = or_gate.output(x)
    print(f'x = {x} ====> {and_gate.output([s1,s2])}')
#======================실습2========================
class perceptron: #perceptron 클래스 구현
    def __init__(self,w):
        self.w = w
    def output(self,x):
        return np.dot(self.w,np.append(1,x))

def sigmoid(x):   #시그모이드 함수
    return 1/(1+np.exp(-x))
def softmax(x): #소프트맥스 함수
    c = np.max(x)
    y = np.exp(x-c)
    sum = np.sum(y)
    return y/sum
def Relu(x): #Relu 함수
    for i in range(len(x)):
        x[i] = max(0,x[i])
    return x

W1 = np.array([[0.1,0.2,0.3],[0.1,0.3,0.5],[0.2,0.4,0.6]]) #w1
W2 = np.array([[0.1,0.2],[0.1,0.4],[0.2,0.5],[0.3,0.6]]) #ㅈ2
Case=[0,0,0,0,0] #case들을 담을 저장공간 (index와맞춰줌)
x = np.array([[1.0],[0.5]]) #입력값

INPUT_LAYER = perceptron(W1.T) #은닉층
OUTPUT_LAYER = perceptron(W2.T) #출력층
sigmoid_input = sigmoid(INPUT_LAYER.output(x)) #시그모이드 함수 은닉층
Relu_input = Relu(INPUT_LAYER.output(x.T)) #Relu함수 은닉층

Case[1] = OUTPUT_LAYER.output(sigmoid_input)
Case[2] = softmax(Case[1])
Case[3] = OUTPUT_LAYER.output(Relu_input)
Case[4] = softmax(Case[3])
print('========2계층 신경망========')
for i in range(1,5):
    print(f"Case {i} ====> {Case[i]}")

#======================실습3========================
current_path = os.path.dirname(os.path.abspath(__file__))
raw_data = pd.read_csv('NN_data.csv',encoding='utf-8', engine = 'python')

X = raw_data[['x0','x1','x2']].to_numpy()    # x 데이터 받기
Y = raw_data['y'].to_numpy() #y 데이터 받기
x0 = raw_data['x0'].to_numpy()  # X0데이터
x1 = raw_data['x1'].to_numpy()  # X1 데이터
x2 = raw_data['x2'].to_numpy()  # X2 데이터
y = raw_data['y'].to_numpy()    #Y 데이터
fig = plt.figure()

ax = fig.add_subplot(projection='3d')   # 3차원 그래프를 그리기 위함
ax.scatter(x0[:301],x1[:301],x2[:301]) #클래스 y값에 따라 3개로 나눔
ax.scatter(x0[301:601],x1[301:601],x2[301:601])
ax.scatter(x0[600:],x1[600:],x2[600:])
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')
plt.show()

def one_hot_encoding(array,size):
    lst = [] #one_hot_encoding 과정
    for i in array:
        tmp = [0] * size
        tmp[i-1] +=1
        lst.append(tmp)
    return np.array(lst)

print('========ONE HOT ENCODING========')
print(one_hot_encoding(Y,3))#3가지케이스로 onehot encoding
class Neural_Network: #2계층 신경망 구현
    def __init__(self,hidden_layer_size,Input_size,Output_size):
        self.hidden_layer_size = hidden_layer_size
        self.Input_size = Input_size
        self.Output_size= Output_size
        self.Create_Weight_Matrix()

    def Create_Weight_Matrix(self): #Weight 를 만드는 함수
        self.W0 = np.random.rand(self.Input_size+1, self.hidden_layer_size)
        self.W1 = np.random.rand(self.hidden_layer_size+1, self.Output_size)
    def Set_Hidden_layer_Node_size(self,size): #히든Node수 설정
        self.hidden_layer_size = size
        self.Create_Weight_Matrix()
    def Check_Input_Output_size(self): #Input,Output 체크함수
        print('Input 속성 수 ====>',self.Input_size)
        print('Output 속성 수 ===>',self.Output_size)
    def predict(self,x): #y예측 함수
        INPUT_LAYER = perceptron(self.W0.T)
        OUTPUT_LAYER = perceptron(self.W1.T)
        sigmoid_input = sigmoid(INPUT_LAYER.output(x))
        return softmax(OUTPUT_LAYER.output(sigmoid_input))

Network = Neural_Network(hidden_layer_size=4,Input_size=3,Output_size=3)
#은닉층 Node size, Input size , Output size 설정
Network.Create_Weight_Matrix()
for i in X:
    print('입력 :',i,'===> 출력 :',Network.predict(i))