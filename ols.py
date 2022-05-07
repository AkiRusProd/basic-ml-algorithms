import random
import matplotlib.pyplot as plt
import numpy as np

""" Ordinary least squares """


w1 = 0.7 #slope steepness
start_w1 = w1 #init value

w2 = random.uniform(-4,4) #movement along the y-axis
start_w2 = w2 #init value

print(f"Initial line: y = {w1} * x + {w2}")

learning_rate = 0.0001
epochs = 3000


std_dev = 10
beta = 10
num = 30


def generate_data(): 
    
    x_data = (np.random.rand(num) * num).round(decimals = 1)
    
    e = (np.random.randn(num) * std_dev).round(decimals = 1)
    
    y_data = x_data * beta + e

    return x_data, y_data

x_data, y_data = generate_data()

bias = 1


# x * w0\
#        > = y # simple perceptron (y = x * w0 + b * w1)
# b * w1/
for i in range (epochs):
    for j in range(len(x_data)):
        x = x_data[j]
        y = w1 * x + w2 * bias
        
        t_y = y_data[j]
        err = - (t_y - y) #MSE derivative

        w1 -= learning_rate * err * x
        w2 -= learning_rate * err * bias

print(f"Line with gradient: y = {w1} * x + {w2}")

w1_f = (num * np.sum(x_data * y_data) - np.sum(x_data) * np.sum(y_data)) / (num * np.sum(x_data * x_data) - np.sum(x_data) * np.sum(x_data))
w2_f = (np.sum(y_data)  - w1_f * np.sum(x_data)) / num


print(f"Line with formula: y = {w1_f} * x + {w2_f}")


x = [i for i in range(0, num)]

def func(k, b):
    return [k * i + b for i in x]

y = func(start_w1, start_w2)

y_network = func(w1, w2)
y_formula = func(w1_f, w2_f)

plt.title("Ordinary least squares")
plt.xlabel("X")
plt.ylabel("Y")

plt.scatter(x_data, y_data, color ='g', s=10, label='Входные данные') 

plt.plot(x, y, 'blue', label='Initial line',)
plt.plot(x, y_network, 'red', label='Gradient descent') 
plt.plot(x, y_formula, 'orange', label='Formula') 

plt.legend(loc=2)

plt.grid(True, linestyle='-', color='0.75')
plt.show()