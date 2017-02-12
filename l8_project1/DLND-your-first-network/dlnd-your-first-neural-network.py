####
# to run interactively:
# % python -i dlnd-your-first-neural-network.py
#
# to open and run pythong file inside interactive python:
# % python
#
# % exec(open('./dlnd-your-first-neural-network.py').read())
# % matplotlib inline // not needed
# % config InlineBackend.figure_format = 'retina' // not needed
#### 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import unittest

###
# load and prepare the data
###
data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
# rides.head() # in interactive

# checking out the data
# rides[:24*10].plot(x='dteday', y='cnt') # in interactive
# plt.show() # in interactive

# dummy variables
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
# data.head() # in interactive

# scaling target variables
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

###
# splitting the data into training, testing and validation sets
###
# save the last 21 days
test_data = data[-21*24:]
data = data[:-21*24]

# separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

# sigmoid function
def sigmoid(x):
    """
    calculate sigmoid
    """
    return 1/(1+np.exp(-x))

###
# time to build the network
###
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in input, hidden and output layers
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                        (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        # set this to your implemented sigmoid function
        # activation function is the sigmoid function
        # self.activation_function = lambda x: return 1/(1+np.exp(-x))
        self.activation_function = sigmoid

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # implement the forward pass here
        # forward pass
        # hidden layer
        # signals into hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
            
        # output layer
        # signals into final output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # signals from final output layer
        final_outputs = final_inputs
            
        # implement the backward pass here
        # backward pass
        # output error
        # output layer error is the difference between desired target and actual output
        output_errors = targets - final_outputs
            
        # backpropagated error
        # errors propagated to the hidden layer
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)
        # hidden layer gradients
        hidden_grad = hidden_outputs * (1 - hidden_outputs)

        # update the weights
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * output_errors * hidden_outputs.T
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_errors * hidden_grad * inputs.T

    def run(self, inputs_list):
        # run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        # implement the forward pass here
        # hidden layer
        # signals into hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # output layer
        # signals into final output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # signals from final output layer
        final_outputs = final_inputs

        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)

###
# training the network
###
# set the hyperparameters here
epochs = 1000
learning_rate = 0.5
hidden_nodes = 12
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    # go through a random batch of 128 records from the training data
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values,
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)

    # printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.ylim(ymax=0.5)


###
# check out your predictions
###
fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)

###
# unit tests
###
inputs = [0.5, -0.2, 0.1]
targets = [0.4]
test_w_i_h = np.array([[0.1, 0.4, -0.3],
                       [-0.2, 0.5, 0.2]])
test_w_h_o = np.array([[0.3, -0.1]])

class TestMethods(unittest.TestCase):
    # unit tests for data loading
    def test_data_path(self):
        # test that file path to dataset has been unaltered
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    def test_data_loaded(self):
        # test that data frame loaded
        self.assertTrue(isinstance(rides, pd.DataFrame))

    # unit tests for network functionality
    def test_activation(self):
        network = NeuralNetwork(3, 2, 1, 0.5)
        # test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, 0.39775194, -0.29887597],
                                              [-0.20185996, 0.50074398, 0.19962801]])))

    def test_run(self):
        # test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[ 0.37275328, -0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[ 0.10562014, 0.39775194, -0.29887597],
                                              [-0.20185996, 0.50074398, 0.19962801]])))

    def test_run(self):
        # test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)






