import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('linear_regression_data01.csv', names=['age', 'tall'])

tall_data = np.asarray(raw_data['tall'].values.tolist())  # 키 데이터(Yn)
age_data = np.asarray(raw_data['age'].values.tolist())  # 나이 데이터(Xn)

#tall_data.sort()  # 데이터 정렬
#age_data.sort()

# 실습1
plt.scatter(age_data, tall_data)  # 데이터 위치를 점으로 표시
plt.title('HomeWork #1')
plt.xlabel('age')
plt.ylabel('tall')
plt.show()

# 실습2
# 강의자료 수식을 코드로 표현
N = len(age_data)
Xn = sum(age_data)/N   # Xn이 두번반복되기에 한번만 계산하도록 저장
W0 = (sum ( tall_data * (age_data-Xn) )/N) / ((sum( age_data**2 ) /N) - Xn ** 2) #분자
W1 = sum( tall_data -  (W0*age_data) ) /N  # 분모
y = W0 * age_data + W1  # 해석해로 구한 선형모델
print('---------------------------------')
print('실습 #2')
print('W0 :', W0)
print('W1 :', W1)
# 실습3
# 해석해로 구한 w0,w1을 이용해 식을 출력
plt.grid(True, linestyle='--')
plt.plot(age_data, y, label='linear-model', color='limegreen')
plt.xlabel('age')
plt.ylabel('tall')
plt.scatter(age_data, tall_data, label='Data')
plt.legend()
plt.title('HomeWork #3')
plt.show()

# 실습4
# MSE구하기
MSE = sum( (y-tall_data)**2 ) / N
print('---------------------------------')
print('실습 #4')
print('MSE :', MSE)

# 실습5
# 머신러닝 교재 6p 식 활용
start_W0, start_W1 = 3, 5  # 시작 w0,w1
cu_W0, cu_W1 = start_W0, start_W1  # 현재 w0,w1 값을 저장할 공간
lr = 0.015  # learning rate
W0s, W1s = [cu_W0], [cu_W1]  # 후에 그래프를 그리기위해 w0,w1를 저장할 공간
MSEs = []  # MSE를 저장해둘 고간
for i in range(5000):  # 5000번 반복
    y_pred = cu_W0 * age_data + cu_W1
    error = tall_data - y_pred

    error_mean = sum(error ** 2) / N  # MSE값

    a_diff = -(1 / N) * sum(age_data * (error))  # 기울기
    b_diff = -(1 / N) * sum((error))  # 기울기값

    cu_W0 = cu_W0 - lr * a_diff
    cu_W1 = cu_W1 - lr * b_diff

    if i % 500 == 0:  # 그래프를 그리기위해 500번 반복시마다 값을 저장
        W0s.append(cu_W0)
        W1s.append(cu_W1)
        MSEs.append(error_mean)


# 실습6
# 저장했던 값들 출력
print('---------------------------------')
print('실습 #6')
print('학습률 : ', lr)
print('초기값 : w0=', start_W0, ', w1 =', start_W1)
print('반복횟수 : 5000')
print('최종 평균제곱오차 :', error_mean)
print('최적 매개변수 : w0 =', cu_W0, ', w1 =', cu_W1)
print('---------------------------------')

# 실습7
plt.grid(True, linestyle='--')
plt.xlabel('step')
plt.plot(range(0, len(W0s) * 500, 500), W0s, label='W1')  # 500번마다 추출했기에 x좌표도 수정을 해주어야 함
plt.plot(range(0, len(W1s) * 500, 500), W1s, label='W0')
plt.scatter(range(0, len(W0s) * 500, 500), W0s)
plt.scatter(range(0, len(W1s) * 500, 500), W1s)
plt.plot(range(500, len(MSEs) * 500, 500), MSEs[1:], label='MSE')
plt.legend()
plt.title('HomeWork #7')
plt.show()

# 실습8
# 해석해로 구한 방법과의 비교
Gradient_descent = cu_W0 * age_data + cu_W1
plt.grid(True, linestyle='--')
plt.plot(age_data, y, label='linear-model', color='limegreen')
plt.xlabel('age')
plt.ylabel('tall')
plt.scatter(age_data, tall_data, label='traning-Data')
plt.plot(age_data, Gradient_descent, label='Gradient_descent', color='blue')
plt.title('HomeWork #8')
plt.legend()
plt.show()


