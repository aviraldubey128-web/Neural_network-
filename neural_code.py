inputs = [1.2, 5.1,2.1,3.5]
weights1 = [1.2, 5.1 ,8.7,2.4]
weights2 = [3.4, -5.7,3.5,-1.4]
weights3 = [-2.4, 4.8,9.1,-5,3]
bias1 = 3
bias2= 4
bias3= 0.4

output = [inputs[0]*weights1[0] + inputs[1]*weights2[1] + inputs[2]*weights3[2] + bias1,
          inputs[0]*weights1[0] + inputs[1]*weights2[1] + inputs[2]*weights3[2] + bias2,
          inputs[0]*weights1[0] + inputs[1]*weights2[1] + inputs[2]*weights3[2] + bias3]

output

output = [-5.52, -4.52, -8.12]


inputs = [1, 2, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2, 3, 0.5]

layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):

    neuron_output = 0

    # For each input and weight
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight

    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)

print(layer_outputs)

output = [4.8, 1.21, 2.385]

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]
biases = [2,3, 0.5]

some_value = -0.5
weights = 0.7
bias = 0.7

some_value*weight
some_value*bias

output = -0.35

'''Tensors, Arrays and Vectors'''
l = [1, 5, 6, 2]
lol = [[1,5,6,2],
       [3,2,1,3]]
lolol = [[[1, 5, 6, 2],
         [3,2,1,3]],
         [[5,2,1,2],
         [64,8,4,0]],
         [[2,8,5,3],
         [1,1,9,4]]]
another_list_of_lists = [[4,2,3],
                         [5,1]]

list_matric_array = [[4,2],
                     [5,1],
                     [8,2]]

print(l)

output: [1, 5, 6, 2]

print(type(l))

output: <class 'list'>

print(np.array(l).shape)

output: (4,)

print(lol)

output: [[1, 5, 6, 2], [3, 2, 1, 3]]

print(type(lol))

output: <class 'list'>

print(np.array(lol).shape)

output: (2, 4)

print(lolol)

output: [[[1, 5, 6, 2], [3, 2, 1, 3]], [[5, 2, 1, 2], [64, 8, 4, 0]], [[2, 8, 5, 3], [1, 1, 9, 4]]]

print(type(lolol))

output: <class 'list'>

print(np.array(lolol).shape)

output: (3, 2, 4) 

print(another_list_of_lists)

output: [[4, 2, 3], [5, 1]]

print(type(another_list_of_lists))

output: <class 'list'>

print(list_matric_array)

output: [[4, 2], [5, 1], [8, 2]]

print(type(list_matric_array))

output: <class 'list'>

print(np.array(list_matric_array).shape)

output: (3, 2)

'''Dot Product and Vector Addition'''

a = [1,2,3]
b = [2,3,4]

dot_product = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
dot_product

output = 20

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

outputs = np.dot(weights, inputs) + bias

outputs

output = '''np.float64(4.8)'''

inputs = [1.0, 2.0, 3.0 ,2.5]
weights = [[0.2, 0.8, -0.8, 1] ,
          [0.5, -0.91, 0.26, -0.5] ,
          [-0.26, -0.27, 0.17, 0.87]]
biases= [2.0, 3.0, 0.5]

layer_outputs = np.dot(weights, inputs) +biases

print(layer_outputs)
np.dot(weights, inputs) == [np.dot(weights[0], inputs),
                        np.dot(weights[1], inputs),
                        np.dot(weights[2], inputs)]

output = '''[3.9   1.21  2.385]
             array([False, False,  True])'''


a = [1,2,3]
np.array([a])

a = [1,2,3]
np.expand_dims(np.array(a), axis=0)

output = '''array([[1, 2, 3]])'''

a = [1,2,3]
b = [2,3,4]

a = np.array([a])
b = np.array([b]).T

np.dot(a,b)

output = '''array([[20]])'''

'''Matrix product'''

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + \
 biases2
print(layer1_outputs)
print(layer2_outputs)

output = '''[[ 4.8    1.21   2.385]
             [ 8.9   -1.81   0.2  ]
             [ 1.41   1.051  0.026]]
             [[ 0.5031  -1.04185 -2.03875]
             [ 0.2434  -2.7332  -5.7633 ]
             [-0.99314  1.41254 -0.35655]]'''



np.random.seed(0)
X = [[1,2,3,2.5],
     [2.0 , 5.0, -1.0, 2.0],
     [-1.5,2.7,3.3,-0.0]]

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    pass
    self.weights = np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)

output = '''[[ 1.07581312 10.39835225  2.44624108  3.1821498   1.88510534]
          [-0.83497962  7.08464114  0.02933567  4.47015253  3.63605378]
          [-4.80938501  6.76410549  0.63465305 -3.23850314 -0.22797236]]
          [[14.82959991 -8.39760189]
          [14.10031472 -1.3404687 ]
          [19.06833489 -8.81538128]]'''


'''!pip install nnfs
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
import random 
'''
X, y =spiral_data(samples=100, classes=3)

plt.scatter(X[:,0], X[:,0])
plt.show()

plt.scatter([X[:,0]], X[:,1], c=y, cmap='brg')
plt.show()


class Layer_Dense:
# Initialize weights and biases
  def __init__(self, inputs):
    pass # using pass statement as aplaceholder

# Forward pass
def forward(self, inputs):
  # Calculate output values from inputs, weights and biases
  pass # using pass statement as a placeholder

  # Layer initialization
  def __init__(self, n__inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
print(np.random.randn(2,5))
print(np.zeros((2,4)))

output: '''[[-1.048553   -1.420018   -1.7062702   1.9507754  -0.5096522 ]
            [-0.4380743  -1.2527953   0.7774904  -1.6138978  -0.21274029]]
            [[0. 0. 0. 0.]
            [0. 0. 0. 0.]]'''

n_inputs = 2
n_neurons = 4

weights = 0.01 * np.random.randn(n_inputs, n_neurons)
biases = np.zeros((1, n_neurons))
print(weights)
print(biases)

output: '''[[-0.00895467  0.00386903 -0.00510805 -0.01180632]
           [-0.00028182  0.00428332  0.00066517  0.00302472]]
           [[0. 0. 0. 0.]]'''

class Layer_Dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward (self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases

X , y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2,3)
dense1.forward(X)
print(dense1.output[:5])

output: '''[[ 0.0000000e+00  0.0000000e+00  0.0000000e+00]
            [ 9.9049321e-05 -2.1095922e-04  3.7016194e-05]
            [ 2.0713788e-04 -4.1655917e-04  7.5199823e-05]
            [ 3.1089142e-04 -6.2469585e-04  1.1282058e-04]
            [ 3.7189163e-04 -8.5043209e-04  1.4422365e-04]]'''





