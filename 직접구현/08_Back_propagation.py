import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"



class perceptron: #perceptron 클래스 구현
    def __init__(self,w):
        self.w = w
    def output(self,x):
        return np.dot(self.w,np.append(x,1))

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
def one_hot_encoding(array,size):
    lst = [] #one_hot_encoding 과정
    for i in array:
        tmp = [0] * size
        tmp[i-1] +=1
        lst.append(tmp)
    return np.array(lst)

class Neural_Network: #2계층 신경망 구현
    def __init__(self,hidden_layer_size,Input,Output,learning_rate,Test_set):
        self.hidden_layer_size = hidden_layer_size
        self.Input_size = Input.shape[1]
        self.Output_size= Output.shape[1]
        self.X = Input
        self.Y = Output
        self.learning_rate = learning_rate
        self.Create_Weight_Matrix()
        self.Test_set = Test_set

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
        self.H = np.append(sigmoid_input,1)
        return sigmoid(OUTPUT_LAYER.output(sigmoid_input))

    def Back_propagation(self):
        lr = self.learning_rate
        for i in range(len(self.X)):

            Y_pred = self.predict(self.X[i])
            Input = np.append(self.X[i], 1)

            for j in range(self.Input_size + 1):  # 역전파 2단계
                for k in range(self.hidden_layer_size):
                    Etotal_h_diff = 0
                    for q in range(self.Output_size):
                        Etotal_h_diff += -2 * (self.Y[i][q] - Y_pred[q]) * Y_pred[q] * (1 - Y_pred[q]) * self.W1[j][q]
                    h_z_diff = self.H[j] * (1 - self.H[j])
                    z_w_diff = Input[j]
                    Etotal_w = Etotal_h_diff * h_z_diff * z_w_diff

                    self.W0[j][k] = self.W0[j][k] - lr * Etotal_w

            for j in range(self.hidden_layer_size+1): #역전파 1단계
                for k in range(self.Output_size):
                    E_o_diff = -2 * ( self.Y[i][k] - Y_pred[k] )
                    o_z_diff = Y_pred[k] * (1 - Y_pred[k])
                    z_w_diff = self.H[j]
                    Etotal_w = E_o_diff * o_z_diff * z_w_diff

                    self.W1[j][k] = self.W1[j][k] - lr * Etotal_w

    def train(self,epoch):
        self.epoch = epoch
        self.MSEs = []
        for i in range(epoch):

            data = np.concatenate([self.X,self.Y],1)  #셔플과정
            np.random.shuffle(data)
            self.X,none,self.Y = np.hsplit(data,(self.Input_size,self.Input_size))

            self.Back_propagation()

            if i % 100 ==0 :

                tmp_mse=[]
                cnt = 0
                for j in range(len(self.X)):
                    Y_pred = self.predict(self.X[j])
                    tmp_mse.append(np.mean((self.Y[j] - Y_pred)**2 ))
                    maxindex = np.argmax(self.predict(self.X[j]))
                    tmp = np.array([0] * self.Output_size)
                    tmp[maxindex] = 1
                    if np.array_equal(tmp,self.Y[j]):
                        cnt +=1
                Accuracy = cnt / len(self.X)

                test_X,none,test_Y = np.hsplit(Test_set,(self.Input_size,self.Input_size))
                cnt = 0
                for j in range(len(test_X)):
                    maxindex = np.argmax(self.predict(test_X[j]))
                    tmp = np.array([0] * self.Output_size)
                    tmp[maxindex] = 1
                    if np.array_equal(tmp,test_Y[j]):
                        cnt +=1

                test_Accuracy = cnt / len(test_X)

                MSE = np.mean(tmp_mse)
                self.MSEs.append(MSE)


                print(f'EPOCH {i} ===> MSE : {MSE} , Accuracy : {Accuracy} , test_Accuracy : {test_Accuracy}')




current_path = os.path.dirname(os.path.abspath(__file__))
raw_data = pd.read_csv('NN_data.csv',encoding='utf-8', engine = 'python')

NN_data = raw_data[['x0','x1','x2','y']].to_numpy()    # x 데이터 받기
NN_data_size = len(NN_data)

X = raw_data[['x0','x1','x2']].to_numpy()    # x 데이터 받기
Y = raw_data['y'].to_numpy() #y 데이터 받기
Y = one_hot_encoding(Y,3)
NN_data = np.concatenate([X,Y],1)

np.random.shuffle(NN_data)
Traning_set = NN_data[:int(0.7*NN_data_size)]
Test_set = NN_data[int(0.7*NN_data_size):]
X,none,Y = np.hsplit(Traning_set,(3,3))

Network = Neural_Network(hidden_layer_size=10,Input=X,Output=Y,learning_rate=0.0001,Test_set=Test_set)
Network.Check_Input_Output_size()

Network.train(10000)
