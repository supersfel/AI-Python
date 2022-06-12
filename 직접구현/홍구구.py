# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 23:51:16 2022

@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dbase = pd.read_csv('NN_data.csv')  # csv 파일 불러오기
data = np.array(dbase)  # 불러온 데이터를 array 형태로 변환하여 data에 저장
data = np.delete(data, 0, axis=1)  # 데이터 활용의 용이함을 위해 기존 데이터의 첫째 열(데이터 순서)을 지움

# 알고리즘 적용 순서 1) 신경망 모델 설계
# N(훈련 데이터 개수) = 630개(900*0.7), M(입력 속성의 개수) = 3개,  L(은닉층 노드의 개수) = 3개, Q(출력 클래스(=노드)의 개수) = 3개
# LR(학습률) = 설정값, epoch = 설정값
N = 630;
M = 3;
L = 3;
Q = 3
LR = 0.0001;
epoch = 2000

# 알고리즘 적용 순서 2) 설계된 모델에 따른 가중치 matrix 생성 및 초기화
w0 = np.random.randn(M + 1, L)
w1 = np.random.randn(L + 1, Q)

# 알고리즘 적용 순서 3) N개의 훈련 데이터 shuffle
# 전체 데이터의 shuffle을 진행(기존 데이터는 유지하며, 셔플된 데이터 생성(permutation))
Shuffle_Data = np.random.permutation(data)

# Shuffle된 데이터 900개를 7:3으로 나누어 훈련 데이터와 검증 데이터로 분류
Train_set = Shuffle_Data[0:630]  # 전체 900개 데이터 중 앞 630개 데이터를 훈련 데이터로 분류
Test_set = Shuffle_Data[630:]  # 전체 900개 데이터 중 뒤 270개 데이터를 검증 데이터로 분류

# 앞서 분류된 데이터를 알고리즘에 활용하기 위해 열 별로 분리(X0~X2, Y 열로 분리하여 저장)
Train_X0 = Train_set[:, 0]
Train_X1 = Train_set[:, 1]
Train_X2 = Train_set[:, 2]
Train_Y = Train_set[:, 3]

print(Train_X0[0],Train_X1[0],Train_X2[0],Train_Y[0])

Test_X0 = Test_set[:, 0]
Test_X1 = Test_set[:, 1]
Test_X2 = Test_set[:, 2]
Test_Y = Test_set[:, 3]


# 알고리즘 적용 순서 4) 0부터 N-1번째 데이터에 대해 순차적으로 오차 역전파 알고리즘을 적용(1 epoch 단위)
# 4-1) 순전파 구현 : n-1에서 업데이트 된 가중치로부터 예측값 y_qn_hat 계산하는 사용자 정의 함수 구성하기
def Sigmoid(z):  # sigmoid 식 구현을 위한 사용자 함수 정의
    return 1 / (1 + np.exp(-z))


def Make_Train_x(i):  # Network의 Training Set 입력 값 정의
    x = [Train_X0[i], Train_X1[i], Train_X2[i]]

    return x


# 은닉층 : Sigmoid 함수에 훈련 데이터의 입력 성분 3가지를 가중치 매트릭스와 크기를 맞추어 연산한 후 입력
# 출력층 : Sigmoid 함수로 설정하여 2계층 신경망을 구현하여 임의로 설정된 가중치 매트릭스로부터 y_hat 값을 도출
def forward_train(i):
    Input_Vector = Make_Train_x(i) + [1]  # 가중치 매트릭스와의 크기를 맞추고자 입력에 바이어스를 붙임

    Hidden_layer_in = np.dot(Input_Vector, w0)  # 입력 벡터와 1번째 가중치 매트릭스 간 연산을 수행한 값을 은닉층의 입력으로 선언
    Hidden_layer_out = Sigmoid(Hidden_layer_in)  # Sigmoid를 통해 은닉층의 출력을 만들어 냄
    Hidden_layer_out_with_bias = np.r_[Hidden_layer_out, [1]]  # 가중치 매트릭스와의 크기를 맞추고자 은닉층 Sigmoid에 따른 출력값에 바이어스를 붙임

    Output_layer_in = np.dot(Hidden_layer_out_with_bias, w1)  # 은닉층의 출력과 2번째 가중치 매트릭스 간 연산을 수행한 값을 출력층의 입력으로 선언
    y_qn_hat = Sigmoid(Output_layer_in)  # 출력층 함수인 Sigmoid 연산을 거쳐 y_qn_hat 값 도출
    # print(i+1, '번째 입력에 따른 y_hat 값', y_qn_hat)
    return y_qn_hat, Hidden_layer_out, Make_Train_x(i)


# 4-2) 훈련 데이터의 Y 값을 One-Hot Encoding을 통해 배열로 변환
total_y = [1, 2, 3]  # y 값의 종류를 total_y로 설정
mapping = {}  # mapping을 위한 dictionary 선언
one_hot_encode = []  # mapping을 통해 1로 변환된 값을 저장할 list 선언

# total_y의 원소 값과 배열 상의 위치 값을 mapping(1→0, 2→1, 3→2)
for x in range(len(total_y)):
    mapping[total_y[x]] = x

# mapping된 y 값에 따라 그 값을 1로 변환하며 y 값이 1인 경우 [1 0 0], 2인 경우 [0 1 0], 3인 경우[0 0 1]의 리스트가 one_hot_encode에 저장됨을 반복한 후
# list 형태의 one_hot_encode를 array 형태로 변환



# 4-3) 실제값(Target, Y)과 예측값(y_qn_hat) 간의 오차로부터 Weight 업데이트 (교재 P.66 ~ 70)
# 교재 P.69 식 23에 따른 w_lq 편미분 MSE 값 계산과 P.70 식 31에 따른 v_ml 편미분 MSE 값 계산을 위한 사용자 정의 함수 선언
# b_ln = 은닉층 출력, e_qn = 예측 오차, 출력층 매개변수(w1) = w_lq, 은닉층 매개변수(w0) = v_ml, 입력값 = x_mn
def part_diff_mse(l, y_qn_hat, b_ln, x_mn):
    one_hot = one_hot_encode[l]
    e_qn = 2 * (y_qn_hat - one_hot) * y_qn_hat * (1 - y_qn_hat)

    w_lq_diff_mse = []
    c = np.r_[b_ln, 1]
    for d in c:
        e = d * e_qn
        w_lq_diff_mse.append(e)

    a = (e_qn * w1).sum(axis=1)  # v_ml_mse 편미분에 사용하기 위한 행렬 a 생성
    b = a[:-1] * b_ln * (1 - b_ln)
    v_ml_diff_mse = []  # a[:-1] 기존 값을 사용하기 위해 바이어스를 붙인 위치의 값을 버리고 계산 진행

    c = np.r_[x_mn, 1]
    for d in c:
        e = d * b
        v_ml_diff_mse.append(e)

    return np.array(w_lq_diff_mse), np.array(v_ml_diff_mse)


# 주어진 n번째 훈련 데이터에 대한 매개변수 업데이트 규칙(교재 P.70 식 32, 33)을 사용자 정의 함수로 선언
def update_weight(w_lq_diff_mse, v_ml_diff_mse):
    global w0, w1

    w0 = w0 - LR * v_ml_diff_mse
    w1 = w1 - LR * w_lq_diff_mse
    return


# 4-4) 입력 630개를 for 문을 통해 알고리즘에 입력 + mse 식 구현을 위한 사용자 함수 정의 : (예측값(y_qn_hat) - 실제 One-hot-Encoding 값)**2.mean()
# + 정확도 판별을 위해 데이터 Y의 One-hot-Encoding 값과 예측값을 비교하기, 일치하면 1, 불일치하면 0을 ACC에 더함
def algorithm(i):
    MSE = 0;
    ACC = 0
    for l in range(N):
        y_qn_hat, b_ln, x_mn = forward_train(l)
        w_lq_diff_mse, v_ml_diff_mse = part_diff_mse(l, y_qn_hat, b_ln, x_mn)
        update_weight(w_lq_diff_mse, v_ml_diff_mse)

        MSE += sum(((y_qn_hat - one_hot_encode[l]) ** 2))
        a = np.argmax(y_qn_hat)
        b = np.argmax(one_hot_encode[l])
        if a == b:
            ACC = ACC + 1
        else:
            ACC = ACC
    return MSE / N, ACC / N


# Training Set에 대한 Network에 따른 MSE와 정확도를 그래프로 나타내기 위해 MSE, ACC, epoch를 담을 리스트를 선언
MSE_list = [];
ACC_list = [];
epoch_list = []

# 알고리즘 적용 순서 5) epoch 수만큼 알고리즘 적용 순서 3~4) 반복(훈련 데이터를 epoch마다 shuffle까지 포함하여 진행)


for n in range(epoch):
    mse, acc_n = algorithm(n)
    MSE_list.append(mse)
    ACC_list.append(acc_n)
    epoch_list.append(n)
    if n % 200 == 0:
        print(n + 1, '번째 epoch에 따른 MSE = ', mse, '정확도 =', acc_n * 100, '%')
# =============================================================================
    Train_set_shuffle = np.random.permutation(Train_set)
#
#     # 앞서 분류된 데이터를 알고리즘에 활용하기 위해 열 별로 분리(X0~X2, Y 열로 분리하여 저장)
    Train_X0 = Train_set_shuffle[:, 0]
    Train_X1 = Train_set_shuffle[:, 1]
    Train_X2 = Train_set_shuffle[:, 2]
    Train_Y = Train_set_shuffle[:, 3]
    one_hot_encode = []
    for c in Train_Y:
        arr = list(np.zeros(len(total_y), dtype=int))
        arr[mapping[c]] = 1
        one_hot_encode.append(arr)
    one_hot_encode = np.array(one_hot_encode)


# =============================================================================

plt.figure('11주차 Training에서의 MSE 및 정확도 변화 그래프')
plt.subplot(211)
plt.plot(epoch_list, MSE_list)
plt.xlabel('Epoch');
plt.ylabel('MSE');
plt.grid(True)  # 그래프의 x, y labels 설정과 grid 표시

plt.subplot(212)
plt.plot(epoch_list, ACC_list)
plt.xlabel('Epoch');
plt.ylabel('ACC');
plt.grid(True)  # 그래프의 x, y labels 설정과 grid 표시


# Test 1) Two-Layer Neural Network “Training”에 의해 나온 가중치 매트릭스 값을 사용하여 Test 진행
# 은닉층 : Sigmoid 함수에 훈련 데이터의 입력 성분 3가지를 가중치 매트릭스와 크기를 맞추어 연산한 후 입력
# 출력층 : Sigmoid 함수로 설정하여 2계층 신경망을 구현하여 Training 결과로 얻은 가중치 매트릭스로부터 y_hat 값을 도출
def Make_Test_x(i):  # Test Set의 데이터를 통해 신경망의 입력을 정의하는 함수 구성
    x = [Test_X0[i], Test_X1[i], Test_X2[i]]
    return x


def forward_test(i):
    Input_Vector = Make_Test_x(i) + [1]  # 가중치 매트릭스와의 크기를 맞추고자 입력에 바이어스를 붙임

    Hidden_layer_in = np.dot(Input_Vector, w0)  # 입력 벡터와 1번째 가중치 매트릭스 간 연산을 수행한 값을 은닉층의 입력으로 선언
    Hidden_layer_out = Sigmoid(Hidden_layer_in)  # Sigmoid를 통해 은닉층의 출력을 만들어 냄
    Hidden_layer_out_with_bias = np.r_[Hidden_layer_out, [1]]  # 가중치 매트릭스와의 크기를 맞추고자 은닉층 Sigmoid에 따른 출력값에 바이어스를 붙임

    Output_layer_in = np.dot(Hidden_layer_out_with_bias, w1)  # 은닉층의 출력과 2번째 가중치 매트릭스 간 연산을 수행한 값을 출력층의 입력으로 선언
    y_qn_hat = Sigmoid(Output_layer_in)  # 출력층 함수인 Sigmoid 연산을 거쳐 y_qn_hat 값 도출
    # print(i+1, '번째 입력에 따른 y_hat 값', y_qn_hat)
    return y_qn_hat


# Test 2) 검증 데이터의 Y 값을 One-Hot Encoding을 통해 배열로 변환
total_y = [1, 2, 3]  # y 값의 종류를 total_y로 설정
mapping = {}  # mapping을 위한 dictionary 선언
one_hot_encode_test = []  # mapping을 통해 1로 변환된 값을 저장할 list 선언

# total_y의 원소 값과 배열 상의 위치 값을 mapping(1→0, 2→1, 3→2)
for o in range(len(total_y)):
    mapping[total_y[o]] = o

# mapping된 y 값에 따라 그 값을 1로 변환하며 y 값이 1인 경우 [1 0 0], 2인 경우 [0 1 0], 3인 경우[0 0 1]의 리스트가 one_hot_encode에 저장됨을 반복한 후
# list 형태의 one_hot_encode를 array 형태로 변환
for p in Test_Y:
    arr = list(np.zeros(len(total_y), dtype=int))
    arr[mapping[p]] = 1
    one_hot_encode_test.append(arr)
one_hot_encode_test = np.array(one_hot_encode_test)

# Test 3) Test Set을 입력한 것에 따른 MSE와 정확도 나타내기
# Test 3-1)Test Set에 대한 알고리즘을 구현해 MSE와 정확도 구하기
MSE_test = 0;
ACC_test = 0
for q in range(len(Test_Y)):
    y_qn_hat = forward_test(q)
    MSE_test += sum(((y_qn_hat - one_hot_encode_test[q]) ** 2))
    a = np.argmax(y_qn_hat)
    b = np.argmax(one_hot_encode_test[q])
    if a == b:
        ACC_test = ACC_test + 1
    else:
        ACC_test = ACC_test

# Test 3-2)Test Set에 대한 알고리즘으로 구한 MSE와 정확도와 Train Set의 마지막 MSE, 정확도를 같이 나타냄으로써 확인
print('Training에 따른 최종 MSE 값 = ', MSE_list[len(epoch_list) - 1])
print('Test Set의 평균 MSE = ', MSE_test / 270)
print('Training에 따른 최종 정확도 = ', ACC_list[len(epoch_list) - 1] * 100, '%')
print('Test Set 대입 결과에 따른 정확도 =', ACC_test / 270 * 100, '%')

# 네트워크 Training까지 완료, Hyper-parameter tuning을 통해 최적화
# 최적화해서 나온 w0, w1에 Test set을 대입을 해서 MSE랑 ACC 구하기(forward만 하면 됨)