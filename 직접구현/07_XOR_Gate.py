import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class perceptron_for_GATE:
    def __init__(self,w):
        self.w = w
    def output(self,x):
        tmp = np.dot(self.w,np.append(1,x))
        result = 1.0*(tmp>0)
        return result
#=========================================실습1=========================================
w_and = np.array([-1.2,1,1])
and_gate = perceptron_for_GATE(w_and)
w_or = np.array([-0.8,1,1])
or_gate = perceptron_for_GATE(w_or)
w_nand = np.array([1.2,-1,-1])
nand_gate = perceptron_for_GATE(w_nand)

x_lst=[[0,0],[1,0],[0,1],[1,1]]
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
for x in x_lst:
    s1 = nand_gate.output(x)
    s2 = or_gate.output(x)
    print(f'x = {x} ====> {and_gate.output([s1,s2])}')
#=========================================실습2=========================================
class perceptron:
    def __init__(self,w):
        self.w = w
    def output(self,x):
        return np.dot(self.w,np.append(1,x))

def sigmoid(x):
    return 1/(1+np.exp(-x))
def softmax(x):
    c = np.max(x)
    y = np.exp(x-c)
    sum = np.sum(y)
    return y/sum
def Relu(x):
    for i in range(len(x)):
        x[i] = max(0,x[i])
    return x

W1 = np.array([[0.1,0.2,0.3],[0.1,0.3,0.5],[0.2,0.4,0.6]])
W2 = np.array([[0.1,0.2],[0.1,0.4],[0.2,0.5],[0.3,0.6]])
Case=[0,0,0,0,0]
x = np.array([[1.0],[0.5]])


INPUT_LAYER = perceptron(W1.T)
OUTPUT_LAYER = perceptron(W2.T)
sigmoid_input = sigmoid(INPUT_LAYER.output(x))
Relu_input = Relu(INPUT_LAYER.output(x.T))

Case[1] = OUTPUT_LAYER.output(sigmoid_input)
Case[2] = softmax(Case[1])
Case[3] = OUTPUT_LAYER.output(Relu_input)
Case[4] = softmax(Case[3])
print('========2계층 신경망========')
for i in range(1,5):
    print(f"Case {i} ====> {Case[i]}")




current_path = os.path.dirname(os.path.abspath(__file__))
raw_data = pd.read_csv('NN_data.csv',encoding='utf-8', engine = 'python')
X = raw_data[['x0','x1','x2']].to_numpy()    # x0 , x1 데이터 받기
Y = raw_data['y'].to_numpy()


def one_hot_encoding(array,size):
    lst = []
    for i in array:
        tmp = [0] * size
        tmp[i-1] +=1
        lst.append(tmp)
    return np.array(lst)

print(one_hot_encoding(Y,3))

x0 = raw_data['x0'].to_numpy()  # X0데이터
x1 = raw_data['x1'].to_numpy()  # X1 데이터
x2 = raw_data['x2'].to_numpy()  # X2 데이터
y = raw_data['y'].to_numpy()    #Y 데이터
N = len(x1)   # N = 80 ( 데이터 수 )
fig = plt.figure()

ax = fig.add_subplot(projection='3d')   # 3차원 그래프를 그리기 위함
ax.scatter(x0[:301],x1[:301],x2[:301])
ax.scatter(x0[301:601],x1[301:601],x2[301:601])
ax.scatter(x0[600:],x1[600:],x2[600:])
ax.set_xlabel('x0')
ax.set_ylabel('x1')
ax.set_zlabel('x2')
plt.show()

class Neural_Network:
    def __init__(self,hidden_layer_size,Input_size,Output_size):
        self.hidden_layer_size = hidden_layer_size
        self.Input_size = Input_size
        self.Output_size= Output_size
        self.Create_Weight_Matrix()

    def Create_Weight_Matrix(self):
        self.W0 = np.random.rand(self.Input_size+1, self.hidden_layer_size)
        self.W1 = np.random.rand(self.hidden_layer_size+1, self.Output_size)
    def Set_Hidden_layer_Node_size(self,size):
        self.hidden_layer_size = size
        self.Create_Weight_Matrix()
    def Check_Input_Output_size(self):
        print('Input 속성 수 ====>',self.Input_size)
        print('Output 속성 수 ===>',self.Output_size)
    def predict(self,x):
        INPUT_LAYER = perceptron(self.W0.T)
        OUTPUT_LAYER = perceptron(self.W1.T)
        sigmoid_input = sigmoid(INPUT_LAYER.output(x))
        return softmax(OUTPUT_LAYER.output(sigmoid_input))

Network = Neural_Network(hidden_layer_size=3,Input_size=3,Output_size=3)
Network.Create_Weight_Matrix()

for i in X:
    print(Network.predict(i))


