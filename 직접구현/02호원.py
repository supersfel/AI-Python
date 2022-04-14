import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('linear_regression_data01.csv', names=['age', 'height'])
# csv파일 불러오기

X = np.asarray(df['age'].values.tolist())  # 배열로 변환
Y = np.asarray(df['height'].values.tolist())  # 배열로 변환
sv = df.sort_values(by='age', ascending=True)  # 나이 오름차순정렬
plt.scatter(X, Y)  # 그래프 출력
plt.grid()
plt.xlabel('age')  # x축이름
plt.ylabel('height')  # y축이름
X_sum = 0  # X값 합
Y_sum = 0  # Y값 합
w0_1 = 0  # w0 분자
w0_2 = 0  # w0 분모
w0 = 0  # w0
w1 = 0  # w1
num1 = 0
num2 = 0
yhat = 0
yhat_x = 0

for n in range(0, 25):
    X_sum += X[n]  # x합 구하기
    Y_sum += Y[n]  # Y합 구하기

X_aver = X_sum / 25  # X평균
Y_aver = Y_sum / 25  # Y평균

for n in range(0, 25):
    w0_1 += Y[n] * (X[n] - X_aver)  # w0 분자 계산
    num1 += X[n] ** 2  # x[n]제곱

num2 = (X_aver) ** 2  # 1/25 * x[n]의 제곱
w0_2 += (num1 * 0.04) - num2  # 분모 계산


w0 = (w0_1 * 0.04) / w0_2
print(w0)