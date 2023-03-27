import numpy as np
import matplotlib.pyplot as plt
import random




def generate_cluster(mean, cov, objects_num):
    return np.random.multivariate_normal(mean, cov, objects_num)

def get_data_with_labels(cluster_1_data, cluster_2_data):
    train_data = []

    for i in range(len(cluster_1_data)):
        train_data.append([cluster_1_data[i][0], cluster_1_data[i][1], 1])
    
    for i in range(len(cluster_2_data)):
        train_data.append([cluster_2_data[i][0], cluster_2_data[i][1], -1])

    return train_data


cluster_1_data = generate_cluster(mean = [2, 2], cov = [[0, 1], [2, 5]], objects_num = 100)
cluster_2_data = generate_cluster(mean = [5, 0], cov = [[4, 2], [2, 2]], objects_num = 100)


train_data = get_data_with_labels(cluster_1_data, cluster_2_data)

train_data = random.sample(train_data, len(train_data))


""" Support Vector Machine """ 

epochs = 500
learning_rate = 0.01
alpha = 0.1

bias = 1

w = np.random.normal(0, 0.05, size = (1, 3))

for epoch in range(epochs):

    for i in range(len(train_data)):
        inputs = np.asfarray(np.concatenate((train_data[i][:2], [bias])))
      
        label = train_data[i][2]

        margin = label * np.dot(w, inputs)
     
        if margin >= 1:
            w -= learning_rate * (alpha * w/epochs)
        else:
            w -= learning_rate * (alpha * w/epochs - inputs * label) 


# x * w0 + y * w1 + w2 = 0 => y = -(x * w0 + w2) / w1

y = lambda x: -(x * w[0][0] + w[0][2]) / w[0][1]


x_disp = np.linspace(np.min(cluster_1_data[:,0]), np.max(cluster_2_data[:,0]), num=10)
y_disp = [y(x) for x in x_disp]



"""Comparison with Gradient descent""" 

w_nn = np.random.normal(0, 0.05, size = (1, 3))


# x * w0\
# y * w1 > = tanh(z) # simple perceptron (z = x * w0 + y * w1 + b * w2)
# b * w2/
for i in range (epochs):
    for j in range(len(train_data)):
        inputs =  np.array(np.concatenate((train_data[j][:2], [bias])), ndmin = 2)
        output = np.tanh(np.dot(inputs, w_nn.T))
        
        target = train_data[j][2]
        err = - (target - output) * (1. - np.power(output, 2)) #MSE derivative

        w_nn -= learning_rate * np.dot(inputs.T, err).T


y_nn = lambda x: -(x * w_nn[0][0] + w_nn[0][2]) / w_nn[0][1]
y_disp_nn = [y_nn(x) for x in x_disp]

plt.title("Support Vector Machine")
plt.xlabel("X")
plt.ylabel("Y")


plt.scatter(cluster_1_data[:,0], cluster_1_data[:,1], marker='_',color='blue', label='cluster 1')
plt.scatter(cluster_2_data[:,0], cluster_2_data[:,1], marker='+',color='green',  label='cluster 2')
 
plt.plot(x_disp, y_disp, 'red', label='SVM') 
plt.plot(x_disp, y_disp_nn, 'orange', label='Gradient descent') 

plt.legend(loc=2)
plt.grid(True, linestyle='-', color='0.75')
plt.show()