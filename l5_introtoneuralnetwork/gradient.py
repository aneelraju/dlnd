
import numpy as np

# defining the sigmoid function for activations
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# the learning rate, eta in the weight step equation
learnrate = 0.5

x = np.array([1,2])
y = np.array(0.5)

# initial weights
w = np.array([0.5, -0.5])

# calculate one gradient descent step for each weight
# the neural network output
nn_output = sigmoid(x[0]*w[0] + x[1]*w[1])
# nn_output = sigmoid(np.dot(x, w))

# calculate error of neural network
error = y - nn_output

# error gradient
error_grad = error * sigmoid_prime(np.dot(x,w))

# calculate change in weights
del_w = [learnrate * error_grad * x[0],
         learnrate * error_grad * x[1]]

print('Neural Network output:')
print(nn_output)
print('Amount of Error:')
print(error)
print('Change in Weights:')
print(del_w)



