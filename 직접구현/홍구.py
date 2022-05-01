import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#실습 1 : 제공된 데이터 파일을 불러들여 x축은 나이, y축은 키를 나타내는 2차원 평면에 각 데이터의 위치를 점으로 표시하라.

dbase = pd.read_csv('lin_regression_data_03.csv', names=['age', 'height'])
                        #csv 파일 불러오기 + 각 데이터를 age, height 필드로 구분하여 데이터를 입력 받아 dbase에 저장
age = dbase['age'].to_numpy()                                       #age의 자료형을 DataFrame에서 numpy_array로 변환
height = dbase['height'].to_numpy()                                 #height의 자료형을 DataFrame에서 numpy_array로 변환
# 실습 5 : 전체 데이터를 차례로 5등분하여 5개의 부분집합으로 나누고, 각 집합의 데이터를 x축은 나이
# y축은 키를 나타내는 2차원 평면에 서로 다른 모양의 마커로 표시하기.
row = 5;
col = 5;
age_slice = [[0 for j in range(col)] for i in range(row)]
age_slice[0] = age[0:5];
age_slice[1] = age[5:10];
age_slice[2] = age[10:15]
age_slice[3] = age[15:20];
age_slice[4] = age[20:25]
# age 배열을 차례로 5등분하여 나누고 순서에 따라 age_slice에 저장

height_slice = [[0 for j in range(col)] for i in range(row)]
height_slice[0] = height[0:5];
height_slice[1] = height[5:10];
height_slice[2] = height[10:15]
height_slice[3] = height[15:20];
height_slice[4] = height[20:25]
# height 배열을 차례로 5등분하여 나누고 순서에 따라 height_slice에 저장

k = [[0 for j in range(col)] for i in range(row)]
k[0] = [age_slice[0], height_slice[0]];
k[1] = [age_slice[1], height_slice[1]];
k[2] = [age_slice[2], height_slice[2]];
k[3] = [age_slice[3], height_slice[3]];
k[4] = [age_slice[4], height_slice[4]];
# 앞서 5등분한 배열들을 k라는 전체 집합의 부분 집합으로 대입

# =============================================================================
# plt.figure('6주차 실습 5번')
# plt.plot(k[0][0], k[0][1], 'b.', k[1][0], k[1][1], 'g.', k[2][0], k[2][1], 'r.', k[3][0], k[3][1], 'c.', k[4][0], k[4][1], 'm.' )
# plt.xlabel('age[months]'); plt.ylabel('height[cm]'); plt.grid(True)
# plt.legend(['0th set', '1st set', '2nd set', '3rd set', '4th set'], loc='upper left')
# =============================================================================

# 실습 6 : 실습 5에서 만든 다섯 개의 데이터 집합을 이용해 5겹 교차검증을 구현(K=9일 때의 가우스 함수를 이용한 선형 기저함수 모델 사용)
# 이를 위해 5개의 홀드아웃 검증을 설계하고 각 홀드아웃의 결과물(매개변수, 일반화 오차) 구하기.
validation = [];
training = []
N = 5


def Q6(N):  # n값에 따라 최적 매개변수를 자동으로 계산하는 함수 선언
    global k
    K = 9
    mean_k = []  # K에 따른 평균값을 담을 배열 mean_k 선언
    Phi_k = []  # K에 따른 Phi 값을 담을 배열 Phi_k 선언
    Phi_random_k = []
    O = np.ones((20, 1))  # 상수 배열 결합을 위해 1로 구성된 배열 O 생성
    O_2 = np.ones((1000, 1))  # 상수 배열 결합을 위해 1로 구성된 배열 O_2 생성
    w_list = []  # w 값을 담기 위한 list로 w_list 선언
    # 전체 배열에서 부분집합 1개를 validation으로 대입하고, 그 외 값들을 training으로 선언하여 홀드아웃 검증을 위한 집단 구성
    # for 문을 통해 훈련 집단과 테스트 집단의 age, height 요소들을 분리하고 배열로 선언하며, 전체 배열에서 제거된 부분집합을
    # validation과 training 집단의 선정이 끝난 후 training 집단 뒤에 붙임으로써 순차적인 집단 구성이 이루어지도록 함
    for n in range(5):

        validation = k[0]
        del k[0]
        training = k
        training_k_age = np.array(list(map(lambda x: x[0], training))).reshape(-1)
        training_k_height = np.array(list(map(lambda x: x[1], training))).reshape(-1)
        k.append(validation)

        a = np.linspace(min(age), max(age), 1000)  # age의 최솟값에서 최댓값 사이에 랜덤으로 1000개의 수 설정
        SD = (max(training_k_age) - min(training_k_age)) / (
                    K - 1)  # 표준 편차에 대한 식 정의(교재 P.17 식 45), SD = Standard Deviation

        for i in range(K):
            mean = min(training_k_age) + SD * i  # 교재 P.17 식 44을 적용하기 위한 k에 따른 평균값 구하는 식 선언
            mean_k.append(mean)  # k에 따른 평균값을 배열 mean_k에 저장
            Phi = np.exp(((-1) / 2) * ((training_k_age - mean_k[i]) / SD) ** 2)  # k에 따른 평균값을 대입하여 Phi 값 구하기
            Phi_random = np.exp(
                ((-1) / 2) * ((a - mean_k[i]) / SD) ** 2)  # y_hat 구하는 것을 용이하게 하기 위해 가중치를 이용하여 그래프를 그리고자 별도의 배열을 통한 식 선언
            Phi_k.append(Phi)  # k에 따른 Phi 값을 배열 Phi_k에 저장
            Phi_random_k.append(Phi_random)
            # print(len(Phi_random_k))
        Phi_kk = np.array(Phi_k)  # Phi_k를 numpy 배열화한 것을 Phi_kk로 설정
        Phi_kk = np.c_[Phi_kk.T, O]  # 위에서 설정한 Phi_kk의 transpose와 상수 1 배열 결합
        Phi_random_kk = np.array(Phi_random_k)
        Phi_random_kk = np.c_[Phi_random_kk.T, O_2]  # 위에서 설정한 Phi_kk의 transpose와 상수 1 배열 결합
        #print(Phi_random_kk.shape)

        height_2 = training_k_height.reshape(20, 1)

        w = np.linalg.pinv(Phi_kk.T @ Phi_kk) @ Phi_kk.T @ height_2  # 다항 차원에서의 해석해 구하기
        w_list.append(w)
        print(w_list)
        # print(Phi_random_kk.shape)
        # y_hat = Phi_random @ w_list[n]                                     #예측치 y_hat을 구하기 위해 가중치와 age의 최소~최대 값 사이의 랜덤 배열을 곱함


Q6(5)