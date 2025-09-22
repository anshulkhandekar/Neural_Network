import numpy as np


inputs = [1.0,2.0,3.0,2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = [2.0]

outputs = np.dot(weights, inputs) + bias

print(outputs)


'''
layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, bias):

    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):

        neuron_output += n_input * weight

    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)

print(layer_outputs)
'''