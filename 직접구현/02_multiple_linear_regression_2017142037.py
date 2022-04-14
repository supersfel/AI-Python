import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
current_path = os.path.dirname(os.path.abspath(__file__))
raw_data = pd.read_csv('multiple_linear_regression_data.csv',encoding='utf-8', engine = 'python')
X = raw_data[['height','weight']].to_numpy()
X = np.c_[X,np.ones(80)]
print(X)
Y = raw_data['label'].to_numpy()
Y = Y.reshape(80,1)


#실습1
height = raw_data['height'].to_numpy()
weight = raw_data['weight'].to_numpy()
age = raw_data['label'].to_numpy()
N = len(height)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(height,weight,age)
#plt.show()

#실습2
analytic_W = np.linalg.pinv(X.T @ X) @ X.T @ Y
height_data = np.linspace(55,190,1000)
weight_data = np.linspace(10,100,1000)
Height,Weight = np.meshgrid(height_data,weight_data)
analytic_y = analytic_W[0]*Height + analytic_W[1]*Weight + analytic_W[2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(height,weight,age)
ax.plot_surface(Height,Weight,analytic_y,cmap='plasma')
ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('age')
ax.set_title('Analytic Solution')
#plt.show()

#실습3

MSE_Analytic_Solution = np.mean( (X @ analytic_W - Y) ** 2 )
print('----------------')
print('실습3')
print('MSE_Analytic_Solution :',MSE_Analytic_Solution)
print('----------------')



#실습4
print('실습4')
def Gradient_Descent(lr,Gradient_W,epoch):
    MSE_Gradient_Descent = np.mean((X @ Gradient_W - age) ** 2)
    print('before MSE_Gradient_Descent :', MSE_Gradient_Descent)
    print('')

    for i in range(epoch):
        y_pred = X @ Gradient_W
        error = y_pred - age


        W0_diff = np.mean((height * error))
        W1_diff = np.mean((weight * error).mean())
        W2_diff = np.mean(error)


        Gradient_W[0] = Gradient_W[0] - lr * W0_diff
        Gradient_W[1] = Gradient_W[1] - lr * W1_diff
        Gradient_W[2] = Gradient_W[2] - lr * W2_diff


        MSE_Gradient_Descent = np.mean((X @ Gradient_W - age) ** 2)
        if i%500 ==0:
            print( "epoch",i, '====> W0 =',Gradient_W[0],'W1 =',Gradient_W[1],'W2 =',Gradient_W[2],'MSE =',MSE_Gradient_Descent)


    print('')
    print('END MSE_Gradient_Descent :', MSE_Gradient_Descent)
    return Gradient_W


Gradient_W = Gradient_Descent(lr=0.00001,Gradient_W=[1,2,-5],epoch=10000)

print('----------------')
print('W0 =',Gradient_W[0],'W1 =',Gradient_W[1],'W2 =',Gradient_W[2])
print('----------------')

gradient_y = Gradient_W[0]*Height + Gradient_W[1]*Weight + Gradient_W[2]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(height,weight,age)
ax.plot_surface(Height,Weight,gradient_y,cmap='plasma')
ax.set_xlabel('height')
ax.set_ylabel('weight')
ax.set_zlabel('age')
ax.set_title('Gradient Desent')
#plt.show()

