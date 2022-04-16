import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv("multiple_linear_regression_data.csv")    # csv 파일 불러오기 


height = dataset['height'].to_numpy()       # 나이 데이터 셋을 배열로 변환 
weight = dataset['weight'].to_numpy()
age = dataset['label'].to_numpy()  


# print(Xn, Xn2, Yn)  

fig = plt.figure()
ax = fig.add_subplot(projection ='3d')

ax.scatter(height, weight, age)

ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('age')

plt.show()

# =============================================================================
#                             실습 1번
# =============================================================================

Xn_matrix = dataset[['height', 'weight']].to_numpy()
Xn_matrix = np.c_[Xn_matrix, np.ones(80)]
Yn_matrix = age.reshape(80,1)

w = np.linalg.pinv(Xn_matrix.T @ Xn_matrix) @ Xn_matrix.T @ Yn_matrix
#print(w)


height_data = np.linspace(55, 190, 1000)
weight_data = np.linspace(10, 100, 1000)



Xn_height, Xn_weight = np.meshgrid(height_data, weight_data)

y_hat = w[0] * Xn_height + w[1] * Xn_weight + w[2]


fig = plt.figure()
ax = fig.add_subplot(projection ='3d')

ax.scatter(height, weight, age)
ax.plot_surface(Xn_height, Xn_weight, y_hat, cmap = 'plasma')

ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('age')

plt.show()


# =============================================================================
#                             실습 2번
# =============================================================================


def MSE(w, Xn_matrix, Yn_matrix):
    loss_func = ((Xn_matrix @ w - Yn_matrix)**2).mean()
    return loss_func

loss_func = MSE(w, Xn_matrix, Yn_matrix)
#print(loss_func)


# =============================================================================
#                             실습 3번
# =============================================================================
# rand_w0 = 3
# rand_w1 = 2.5
# rand_w2 = 4

# learning_rate = 0.5

# num_epoch = 50000

# error_list = []
# rand_w0_list = []
# rand_w1_list = []
# rand_w2_list = []

# def diff_w0_MSE(rand_w0_list, rand_w1_list, rand_w2_list, height, weight, age):
#     num = 0
#     N = len(height)
#     for n in range(N):
#         num += height[n]* (rand_w0 * height[n] + rand_w1 * weight[n] + rand_w2 - age[n])
#     result = (2 / N) * num 
#     return result

# def diff_w1_MSE(rand_w0_list, rand_w1_list, rand_w2_list, height, weight, age):
#     num = 0
#     N = len(height)
#     for n in range(N):
#         num += weight[n]* (rand_w0 * height[n] + rand_w1 * weight[n] + rand_w2 - age[n])
#     result = (2 / N) * num 
#     return result

# def diff_w2_MSE(rand_w0_list, rand_w1_list, rand_w2_list, height, weight, age):
#     num = 0
#     N = len(height)
#     for n in range(N):
#         num += (rand_w0 * height[n] + rand_w1 * weight[n] + rand_w2 - age[n])
#     result = (2 / N) * num 
#     return result

# w0_MSE_diff = diff_w0_MSE(rand_w0_list, rand_w1_list, rand_w2_list, height, weight, age)
# w1_MSE_diff = diff_w1_MSE(rand_w0_list, rand_w1_list, rand_w2_list, height, weight, age)
# w2_MSE_diff = diff_w2_MSE(rand_w0_list, rand_w1_list, rand_w2_list, height, weight, age)

# for i in range(num_epoch):
#     Y_pred = rand_w0 * height + rand_w1 * weight + rand_w2
    
#     error = ((Y_pred - y_hat) ** 2).mean()
    
#     rand_w0 = rand_w0 - learning_rate * ((Y_pred - y_hat) * height).mean()
#     rand_w1 = rand_w1 - learning_rate * ((Y_pred - y_hat) * weight).mean()
#     rand_w2 = rand_w2 - learning_rate * (Y_pred - y_hat).mean()
    
#     before_error = error
    
#     if abs(before_error - error) < 0.01:
#         print(rand_w0)
#         print(rand_w1)
#         print(rand_w2)
#         print(error)
#         break
    
    
#     rand_w0_list.append(rand_w0)
#     rand_w1_list.append(rand_w0)
#     rand_w2_list.append(rand_w0)
#     error_list.append(error)

# =============================================================================
#                             실습 4번
# =============================================================================
   
dataset = pd.read_csv("linear_regression_data01.csv"     # csv 파일 불러오기
                      ,names = ['age' , 'height'])     

x_data = dataset['age']              # 나이 데이터 셋
y_data = dataset['height']           # 키  데이터 셋

Xn = dataset['age'].to_numpy()       # 나이 데이터 셋을 배열로 변환 
Yn = dataset['height'].to_numpy()    # 키 데이터 셋을 배열로 변환


def Gaussian(Xn, Yn, K):
    Uk_list = []
    temp = []
 
    Yn_matrix = Yn.reshape(25,1)
    
    min_Xn = min(Xn)
    max_Xn = max(Xn)
    for k in range(0, K):
        Uk = min_Xn + ((max_Xn - min_Xn) /( K - 1)) * k
        Uk_list.append(Uk) 
    
    sigma = (max_Xn - min_Xn) / (  K - 1 )
    print(K,'sigma',sigma)
    Gauss_x = 0    
    for k in range(0, K):
        Gauss_x = np.exp(-0.5 * (((Xn - Uk_list[k]) / sigma) ** 2))                        
        temp.append(Gauss_x)    # 

    
    Gauss_x_list = np.array(temp) 
    #print(Gauss_x_list.shape)# 3 x 25
    # print(Gauss_x_list)
    #print('K',K)
    #print('Gauss_x_list',Gauss_x_list)
    
    T_Gauss_x_list = np.transpose(Gauss_x_list)   # 25 x 3
    print('T_Gauss_x_list',T_Gauss_x_list)
    print(T_Gauss_x_list.shape)
    Gauss_x_matrix = np.c_[T_Gauss_x_list, np.ones(25)]   # 25 x 4
    w = np.linalg.pinv(Gauss_x_matrix.T @ Gauss_x_matrix) @ Gauss_x_matrix.T @ Yn_matrix   # inv(4 x 25 @ 25 x 4) @ 4 x 25 @ 25 x 1
    
    #print(w) # 결과물 : 1 x 4
    
    y_hat = Gauss_x_matrix @ w
    
    if K == 3:
        plt.title('K = 3')
    elif K == 5:
        plt.title('K = 5')
    elif K == 8:
        plt.title('K = 8')
    elif K == 10:
        plt.title('K = 10')
    plt.xlabel('age')
    plt.ylabel('height')
    plt.scatter(x_data, y_data, color = 'blue', label = 'training_data')
    plt.scatter(Xn, y_hat, color = 'red', label = 'Gaussian_regression')
    plt.grid()
    plt.legend()
    plt.show()
    
  
    
    return y_hat

K3_y_hat = Gaussian(Xn, Yn, 3)
K5_y_hat = Gaussian(Xn, Yn, 5) 
K8_y_hat = Gaussian(Xn, Yn, 8)
K10_y_hat = Gaussian(Xn,Yn, 10)




# # =============================================================================
# #                             실습 5번, 실습 6번
# # ============================================================================= 

K = [3, 5, 8, 10]
MSE_list = []

K3_MSE = ((K3_y_hat - Yn)**2).mean()
MSE_list.append(K3_MSE)

K5_MSE = ((K5_y_hat - Yn)**2).mean()
MSE_list.append(K5_MSE)

K8_MSE = ((K8_y_hat - Yn)**2).mean()
MSE_list.append(K8_MSE)

K10_MSE = ((K10_y_hat - Yn)**2).mean()
MSE_list.append(K10_MSE)
      
plt.xlabel('K')
plt.ylabel('MSE')
plt.grid()
plt.legend()
plt.scatter(K, MSE_list , label = 'MSE')
plt.show()
  