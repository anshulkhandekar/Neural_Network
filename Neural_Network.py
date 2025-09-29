# Importing libraries
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
nnfs.init()

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

class Activation_RelU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
activation1 = Activation_RelU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

# plt.scatter(X[:,0], X[:,1],c=y, cmap='brg')
# plt.show()



# Manual creation of inputs, weights, & biases
# inputs = [[1.0,2.0,3.0,2.5],
#         [2.0, 5.0, -1.0, 2.0], 
#          [-1.5, 2.7, 3.3, -0.8]]
# weights = [[0.2, 0.8, -0.5, 1.0],
#          [0.5, -0.91, 0.26, -0.5],
#         [-0.26, -0.27, 0.17, 0.87]]
# biases = [2.0, 3.0, 0.5]
# weights2 = [[0.1, -0.14, 0.5],
#            [-0.5, 0.12, -0.33],
#            [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]

# Feeding inputs to first layer and then resulting outputs to second layer
# layer_outputs1 = np.dot(inputs, np.array(weights).T) + biases
# layer_outputs2 = np.dot(layer_outputs1, np.array(weights2).T) + biases2

# print(layer_outputs2)
