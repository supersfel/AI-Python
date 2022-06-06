import pandas as pd
import numpy as np
import os  # 현재 디렉토리 확인하기
import matplotlib.pyplot as plt
import copy

current_path = os.path.dirname(os.path.abspath(__file__))  # os.path.dirname : 경로 중 디렉토리명만 얻기
raw_data = pd.read_csv(current_path + '/NN_data.csv', encoding='utf-8', engine='python')

x0 = raw_data['x0'].to_numpy()  # NN_data.csv 900개의 x0값을 불러오난 코드
x1 = raw_data['x1'].to_numpy()  # NN_data.csv 900개의 x1값을 불러오난 코드
x2 = raw_data['x2'].to_numpy()  # NN_data.csv 900개의 x2값을 불러오난 코드
y = raw_data['y'].to_numpy()  # NN_data.csv 900개의 y값을 불러오난 코드

x_data = raw_data[['x0', 'x1', 'x2']].to_numpy()  # 입력값들만 모아준다.
y_list = copy.deepcopy(y)
y = y.tolist()  # tolist 배열로 변환

class_group = list(set(y))  # 분류 할 class를 세기 위해 중복되는 값을 제거해 준다.


def make_onehot(n):  # one_hot encoding을 구현하는 코드
    one_hot = []
    for i in range(len(class_group)):
        one_hot.append(0)
    one_hot[n - 1] = 1
    return one_hot


class1_onehot = make_onehot(1)  # class가 1일시 [1, 0, 0]을 만들어 준다.
class2_onehot = make_onehot(2)  # class가 2일시 [0, 1, 0]을 만들어 준다.
class3_onehot = make_onehot(3)  # class가 3일시 [0, 0, 1]을 만들어 준다.

for i in range(len(y)):
    if y[i] == 1:
        y[i] = class1_onehot
    elif y[i] == 2:
        y[i] = class2_onehot
    else:
        y[i] = class3_onehot

y = np.array(y)

randomize = np.arange(len(x_data))
np.random.shuffle(randomize)

x_shuffle = x_data[randomize]  # x_data 900개를 shuffle
y_shuffle = y[randomize]  # y_data 900개를 shuffle

x_training = x_shuffle[:630]  # x_data의 shuffle된 900개의 데이터 중 630개의 training set분류
y_training = y_shuffle[:630]  # y값의 shuffle된 900개의 데이터 중 630개의 training set분류

x_test = x_shuffle[630:]  # x의 test set 분류
y_test = y_shuffle[630:]  # y의 test set 분류
y_test_index = np.argmax(y_test, axis=1)  # y의 test set의 가장 큰값의 index 추출


def make_w0(M):
    w0 = np.random.randn(M + 1, L)  # 평균이0, 표준편차가 1인 가우시안 표준정규분포 난수
    return w0  # 첫번째 weight를 (입력 속성 수+1) x (은닉층의 노드수)크기의 임의의 weight값을 갖는 Matrix 생성


def make_w1(L):
    w1 = np.random.randn(L + 1, Q)  # 평균이0, 표준편차가 1인 가우시안 표준정규분포 난수
    return w1  # 두번쨰 weight를 (은닉층 노드수+1) x (출력 클래스의 노드 수)크기의 임의의 weight값을 갖는 Matrix 생성


def sigmoid_hidden(lst):  # 들어오는 list를 softmax함수로 반환
    hidden_output = copy.deepcopy(lst)
    h_x = 1 / (1 + np.exp(-hidden_output))
    return h_x


def sigmoid_output(lst):  # 들어오는 list를 Relu함수로 반환
    output = copy.deepcopy(lst)
    o_x = 1 / (1 + np.exp(-output))
    return o_x


def check_index(lst):  # 최종적으로 나온 y의 값에서 class 분류해주는 함
    data = lst
    index_list = np.argmax(data, axis=1) + 1
    return index_list


N = len(x_training)  # 훈련 데이터 갯수
M = 3  # 입력 속성의 개수
L = 6  # Hidden Layer의 노드 수를 자유롭게 설정하는 기능
Q = 3  # Output Class 수를 정할 수 있는 기능
LR = 0.0001

w0 = make_w0(M)  # 임의의 w0 생성
w0 = np.array(w0)
w1 = make_w1(L)  # 임의의 w1생성
w1 = np.array(w1)

print("초기 w0값 : ", w0)
print('\n')
print('초기 w1값 : ', w1)
print('\n')

MSE = []  # 1 Epoch가 진행되면 저장되는 MSE


def forward(n):
    X_data = np.c_[x_training, np.ones(N)]  # 입력 속성에 bias 추
    hidden_data = X_data[n] @ w0
    hidden_layer = sigmoid_hidden(hidden_data)  # 은닉층 활성화 함수 통과
    hidden_layer_b = np.r_[hidden_layer, np.ones(1)]  # 은닉층의 활성화 함수 통과 후 bias 추가
    output_data = hidden_layer_b @ w1
    result = sigmoid_output(output_data)

    return X_data, result, hidden_layer_b, hidden_layer, w0, w1


def backward(n):
    x_list, est, hid_layer_b, hid_layer, w0, w1 = forward(n)  # Forward propagation에서 구한 data들을 가져옴

    if n == 629:
        mse_hid = x_list @ w0
        mse_hid_layer = sigmoid_hidden(mse_hid)
        mse_hid_bias = np.c_[mse_hid_layer, np.ones(len(x_list))]
        mse_output_data = mse_hid_bias @ w1
        mse_result = sigmoid_output(mse_output_data)
        MSE.append((np.sum(mse_result - y_training) ** 2) / len(mse_result))  # 1 Epoch 진행 후 MSE값을 저장

    error = y_training[n] - est
    first_do = -2 * error * est * (1 - est)  # (E_total / o1)미분 X (o / z)미분
    hid_layer_b = np.reshape(hid_layer_b, (L + 1, 1))
    first_do = np.reshape(first_do, (1, Q))  # -> 역전파 1단계 구하는 과정


    diff_w1 = hid_layer_b @ first_do  # w1의 편미분 식

    x_input = x_list[n]
    x_input = np.reshape(x_input, (4, 1))

    mid_etotal = -2 * (y_training[n] - est) * est * (1 - est)
    mid_etotal = np.reshape(mid_etotal, (1, 3))

    mid_etotal_re = np.reshape(mid_etotal, (3, 1))
    etotal_hid = w1 @ mid_etotal_re  # 역전파 2단계 구하는 과정
    etotal_hid = np.delete(etotal_hid, (L), axis=0)  # 은닉층의 마지막 h는 입력에서의 bias의 영향을 받지않기 때문에 삭제
    hid_layer = np.reshape(hid_layer, (6, 1))

    second_do = etotal_hid * hid_layer * (1 - hid_layer)
    second_do = np.reshape(second_do, (1, L))
    diff_w0 = x_input @ second_do  # w0의 편미분 식


    w1 = w1 - LR * diff_w1  # w1 update
    w0 = w0 - LR * diff_w0  # w0 update
    return w0, w1  # w0와 w1을 return해 줌으로써 계속 최신화 시켜준다.


def percent():  # test set 270개의 Accuracy 구하는 함수

    x_testset = np.c_[x_test, np.ones(len(x_test))]
    middle_data = x_testset @ w0
    middle_layer = sigmoid_hidden(middle_data)
    middle_layer_b = np.c_[middle_layer, np.ones(len(x_testset))]
    output = middle_layer_b @ w1
    result = sigmoid_output(output)  # test set 270개를 Forward propagation해서 y값 도출

    sum = 0
    index = np.argmax(result, axis=1)

    for i in range(len(x_testset)):  # test set의 예측값과 실제값의 최대값 index가 일치하는지 확인
        if (index[i] == y_test_index[i]):
            sum += 1
    accuracy = sum / len(x_testset) * 100
    return accuracy


for i in range(2001):  # 200 Epoch마다 Accuracy 출력
    for j in range(630):
        forward(j)
        w0, w1 = backward(j)
    if i % 200 == 0:
        Accuracy = percent()
        print('Epoch : {} ==============> Accuracy : {}%'.format(i, Accuracy))

print('최종 w0 : ', w0)  # 2001 Epoch 후 updata된 최종 w0
print('최종 w1 : ', w1)  # 2001 Epoch 후 updata된 최종 w1

MSE_size = range(len(MSE))

plt.xlabel('Epoch')
plt.ylabel("MSE")
plt.grid()
plt.xlim(0, 2000)
plt.ylim(0, 30)
plt.plot(MSE_size, MSE, color='blue')
plt.show()
# MSE의 변화율을 보는 코드a