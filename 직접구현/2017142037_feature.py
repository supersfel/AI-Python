import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
class perceptron:  # perceptron 클래스 구현
    def __init__(self, w):
        self.w = w
    def output(self, x):
        return np.dot(np.append(x, 1), self.w)
def sigmoid(x):  # 시그모이드 함수
    return 1 / (1 + np.exp(-x))
def one_hot_encoding(array, size):
    lst = []  # one_hot_encoding 과정
    for i in array:
        tmp = [0] * size
        tmp[i - 1] += 1
        lst.append(tmp)
    return np.array(lst)
class Neural_Network:  # 2계층 신경망 구현
    def __init__(self, hidden_layer_size, Input, Output, learning_rate,Test_set):
        self.hidden_layer_size = hidden_layer_size  # 은닉층 노드 개수
        self.Input_size = Input.shape[1]  # 입력층 사이즈
        self.Output_size = Output.shape[1]  # 출력층 사이즈
        self.X = Input  # 트레이닝 셋 입력
        self.Y = Output  # 트레이닝 셋 출력(정답)
        self.learning_rate = learning_rate  # Learning rate
        self.Create_Weight_Matrix()  # W0,W1매트릭스 임의 생성
        self.Test_set = Test_set  # 테스트셋 저장

    def Create_Weight_Matrix(self):  # Weight 를 만드는 함수
        self.W0 = np.random.randn(self.Input_size + 1, self.hidden_layer_size)
        self.W1 = np.random.randn(self.hidden_layer_size + 1, self.Output_size)

    def Set_Hidden_layer_Node_size(self, size):  # 히든Node수 설정
        self.hidden_layer_size = size
        self.Create_Weight_Matrix()

    def Check_Input_Output_size(self):  # Input,Output 체크함수
        print('Input 속성 수 ====>', self.Input_size)
        print('Output 속성 수 ===>', self.Output_size)

    def predict(self, x):  # y예측 함수
        INPUT_LAYER = perceptron(self.W0)
        OUTPUT_LAYER = perceptron(self.W1)
        self.sigmoid_input = sigmoid(INPUT_LAYER.output(x))
        self.H = np.append(self.sigmoid_input, 1)
        return sigmoid(OUTPUT_LAYER.output(self.sigmoid_input))

    def Back_propagation(self):
        lr = self.learning_rate  # Learning rate
        for i in range(len(self.X)):
            Y_pred = self.predict(self.X[i])  # Y 예측값
            Input = np.append(self.X[i], 1)  # 입력층 + 1 추가
            # 역전파를 1단계부터 시행하면 W가 업데이트되어 2단계부터 시행
            for j in range(self.Input_size + 1):  # 역전파 2단계
                for k in range(self.hidden_layer_size):
                    Etotal_h_diff = 0
                    for q in range(self.Output_size):
                        Etotal_h_diff += -2 * (self.Y[i][q] - Y_pred[q]) * Y_pred[q] * (1 - Y_pred[q]) * self.W1[k][q]
                    h_z_diff = self.H[k] * (1 - self.H[k])
                    z_w_diff = Input[j]
                    Etotal_w = Etotal_h_diff * h_z_diff * z_w_diff
                    self.W0[j][k] = self.W0[j][k] - lr * Etotal_w  # W0업데이트


            for j in range(self.hidden_layer_size + 1):  # 역전파 1단계
                for k in range(self.Output_size):
                    E_o_diff = -2 * (self.Y[i][k] - Y_pred[k])
                    o_z_diff = Y_pred[k] * (1 - Y_pred[k])
                    z_w_diff = self.H[j]
                    Etotal_w = E_o_diff * o_z_diff * z_w_diff
                    self.W1[j][k] = self.W1[j][k] - lr * Etotal_w  # W1 업데이트

    def train(self, epoch):
        self.epoch = epoch  # epoch 저장
        self.MSEs = []  # MSE그래프를 그리기 위함
        self.Accuaracys = []  # 정확도 그래프를 그리기 위함
        for i in range(epoch):
            data = np.concatenate([self.X, self.Y], 1)  # 셔플과정
            np.random.shuffle(data)  # 매 에폭마다 섞어주는 과정
            self.X, none, self.Y = np.hsplit(data, (self.Input_size, self.Input_size))
            self.Back_propagation()  # 역전파 과정으로 W업데이트


            if i % 100 == 0:
                tmp_mse = []  # 정확도 계산
                cnt = 0
                for j in range(len(self.X)):
                    Y_pred = self.predict(self.X[j])
                    tmp_mse.append(np.mean((self.Y[j] - Y_pred) ** 2))
                    maxindex = np.argmax(self.predict(self.X[j]))  # 가장큰 index가져오기
                    tmp = np.array([0] * self.Output_size)
                    tmp[maxindex] = 1
                    if np.array_equal(tmp, self.Y[j]):  # 정답과 비교
                        cnt += 1
                Accuracy = cnt / len(self.X)
                self.Accuaracys.append(Accuracy)
                MSE = np.mean(tmp_mse)
                self.MSEs.append(MSE)

                test_X, none, test_Y = np.hsplit(Test_set, (self.Input_size, self.Input_size))
                cnt = 0
                for j in range(len(test_X)):
                    maxindex = np.argmax(self.predict(test_X[j]))
                    tmp = np.array([0] * self.Output_size)
                    tmp[maxindex] = 1
                    if np.array_equal(tmp, test_Y[j]):
                        cnt += 1
                test_Accuracy = cnt / len(test_X)

                print(f'EPOCH {i} ===> MSE : {MSE} , Accuracy : {Accuracy} ,Test_Accuracy : {test_Accuracy}')

                # W 저장 ( W폴더를 하나 만들어야 저장이 가능함 )
                # df0 = pd.DataFrame(self.W0)
                # df0.to_csv(f'W\\{i}epoch_W0.csv',index=False,header='None')
                # df1 = pd.DataFrame(self.W1)
                # df1.to_csv(f'W\\{i}epoch_W1.csv', index=False,header='None')



#####################################################################################

def feature_1(input_data):
    # 특징 후보 1번 : 가로축 Projection => 확률밀도함수로 변환 => 기댓값
    X = sum(input_data)
    input_data = input_data.T

    S = sum(X)
    pdf = np.array([sum(input_data[i]) / S for i in range(len(input_data))])
    output_value = sum(X * pdf)
    return output_value

#####################################################################################
def feature_2(input_data):
    # 특징 후보 2번 : 가로축 Projection => 확률밀도함수로 변환 => 분산
    X = sum(input_data)
    input_data = input_data.T

    S = sum(X)
    pdf = np.array([ sum(input_data[i])/S for i in range(len(input_data)) ])
    E = sum(X * pdf)
    output_value = sum( (X - E)**2 * pdf)
    return output_value
#####################################################################################
def feature_3(input_data):
    # 특징 후보 3번 : 세로축 Projection => 확률밀도함수로 변환 => 기댓값
    X = sum(input_data.T)
    S = sum(X)
    pdf = np.array([sum(input_data[i]) / S for i in range(len(input_data))])
    output_value = sum(X * pdf)
    return output_value
#####################################################################################
def feature_4(input_data):
    # 특징 후보 4번 : 세로축 Projection => 확률밀도함수로 변환 => 분산
    X = sum(input_data.T)
    input_data = input_data

    S = sum(X)
    pdf = np.array([sum(input_data[i]) / S for i in range(len(input_data))])
    E = sum(X * pdf)
    output_value = sum((X - E) ** 2 * pdf)
    return output_value
def feature_5(input_data):
    # 특징 후보 5번 : Diagonal 원소배열 추출 => 밀도함수로 변환 => 기댓값
    input_data = np.diag(input_data.T)
    S = sum(input_data)
    pdf = np.array([input_data[i] / S for i in range(len(input_data))])
    output_value = sum(input_data * pdf)
    return output_value
def feature_6(input_data):
    # 특징 후보 6번 : Diagonal 원소배열 추출 => 밀도함수로 변환 => 분산
    input_data = np.diag(input_data.T)
    S = sum(input_data)
    pdf = np.array([input_data[i] / S for i in range(len(input_data))])
    E = sum(input_data * pdf)
    output_value = sum((input_data - E) ** 2 * pdf)
    return output_value
def feature_7(input_data):
    # 특징 후보 7번 : Diagonal 원소배열 추출 => 0의 개수
    input_data = np.diag(input_data.T)
    cnt=0
    for i in input_data:
        if i == 0:
            cnt +=1
    return cnt
def feature_8(input_data):
    # 특징 후보 8번 : Anti-Diagonal 원소배열 추출 => 밀도함수로 변환 => 기댓값
    input_data = np.diag(np.fliplr(input_data))
    S = sum(input_data)
    pdf = np.array([input_data[i] / S for i in range(len(input_data))])
    output_value = sum(input_data * pdf)
    return output_value

def feature_9(input_data):
    # 특징 후보 9번 : Anti-Diagonal 원소배열 추출 => 밀도함수로 변환 => 분산
    input_data = np.diag(np.fliplr(input_data))
    S = sum(input_data)
    pdf = np.array([input_data[i] / S for i in range(len(input_data))])
    E = sum(input_data * pdf)
    output_value = sum((input_data - E) ** 2 * pdf)
    return output_value
def feature_10(input_data):
    # 특징 후보 10번 : Anti-Diagonal 원소배열 추출 => 0의 개수
    input_data = np.diag(np.fliplr(input_data))
    cnt=0
    for i in input_data:
        if i == 0:
            cnt +=1
    return cnt

feature = [0,feature_1,feature_2,feature_3,feature_4,feature_5,feature_6,feature_7,feature_8,feature_9,feature_10]

Training_X = np.array([],dtype='float32')  #입력 데이터 가공
Training_X=np.resize(Training_X,(0,5))
Training_Y = np.array([],dtype='float32')
Training_Y = np.resize(Training_X,(0,3))
for i in range(3):
    for j in range(1,501):
        tmp_name = f'[배포용] MINIST Data\\{i}_{j}.csv'
        tmp_img = pd.read_csv(tmp_name,header=None).to_numpy(dtype='float32')

        x0 = feature[2](tmp_img)
        x1 = feature[4](tmp_img)
        x2 = feature[6](tmp_img)
        x3 = feature[7](tmp_img)
        x4 = feature[9](tmp_img)

        X = np.array([[x0,x1,x2,x3,x4]],dtype='float32')
        Y = np.array([[0] * 3])
        Y[0][i] = 1
        Training_X = np.concatenate((Training_X,X),axis=0)
        Training_Y = np.concatenate((Training_Y,Y),axis=0)
#데이터 셔플 및 Train,Test set 구분
Train_data = np.concatenate([Training_X,Training_Y],1)
np.random.shuffle(Train_data)
Traning_set = Train_data[:1200]
Test_set = Train_data[1200:]
X,none,Y = np.hsplit(Traning_set,(5,5))

Network = Neural_Network(hidden_layer_size=4,Input=X,Output=Y,learning_rate=0.007,Test_set=Test_set)
Network.Check_Input_Output_size()

Network.train(5000)
print('W0 :',Network.W0) #학습된 W0,W1 출력
print('W1 :',Network.W1)

plt.xlabel('Epoch')
plt.ylabel("MSE")
plt.grid()
plt.plot(range(len(Network.MSEs)),Network.MSEs,color='blue')
plt.show() #MSE 변화 출력

plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.grid()
plt.plot(range(len(Network.Accuaracys)),Network.Accuaracys,color='blue')
plt.show() #정확도 변화 출력

