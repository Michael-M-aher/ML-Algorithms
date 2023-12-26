import numpy as np

class Neuron:
    '''
    Neuron class

    Attributes:
        weights (numpy array): weights of the neuron
        activation (function): activation function of the neuron
        bias (float): bias of the neuron
        error (float): error of the neuron
        output (float): output of the neuron

    Methods:
        _feed_forward: feed forward the input to the neuron
        _backpropagation: backpropagate the error to the previous layer
    '''
    def __init__(self, weights, activation):
        '''
        Constructor for Neuron class

        Parameters:
            weights (numpy array): weights of the neuron
            activation (function): activation function of the neuron

        Returns:
            None
        '''
        self.weights = weights
        self.activation = activation
        self.bias = 0
        self.error = 0
        self.output = 0

    def _feed_forward(self, inputs):
        '''
        Feed forward the input to the neuron

        Parameters:
            inputs (numpy array): input to the neuron

        Returns:
            output (float): output of the neuron
        '''
        total = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation(total) if self.activation else total
        return self.output
    
    def _backpropagation(self, output_error, next_layer, idx):
        '''
        Backpropagate the error to the previous layer

        Parameters:
            output_error (float): error of the output layer
            next_layer (Layer): next layer of the neuron
            idx (int): index of the neuron in the next layer

        Returns:
            error (float): error of the neuron
        '''
        mul1 = 0
        for neuron in next_layer.neurons:
            mul1 += neuron.weights[idx] * output_error
        return mul1 * self.output * (1 - self.output)

    
class Layer:
    '''
    Layer class

    Attributes:
        neurons (list): list of neurons in the layer
        input_size (int): input size of the layer
        output_size (int): output size of the layer
        activation (function): activation function of the layer
        output (numpy array): output of the layer

    Methods:
        _feed_forward: feed forward the input to the layer
        _backpropagation: backpropagate the error to the previous layer
        _update_weights: update the weights of the layer
        _update_bias: update the bias of the layer
    '''
    def __init__(self, input_size, output_size, activation=None):
        '''
        Constructor for Layer class

        Parameters:
            input_size (int): input size of the layer
            output_size (int): output size of the layer
            activation (function): activation function of the layer

        Returns:
            None
        '''
        self.activation = activation if activation else linear
        self.neurons = []
        self.input_size = input_size
        self.output_size = output_size

    def _feed_forward(self, inputs):
        '''
        Feed forward the input to the layer

        Parameters:
            inputs (numpy array): input to the layer

        Returns:
            outputs (numpy array): outputs of the layer
        '''
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron._feed_forward(inputs))
        return np.array(outputs)
    
    def _backpropagation(self, output_error, next_layer):
        '''
        Backpropagate the error to the previous layer

        Parameters:
            output_error (float): error of the output layer
            next_layer (Layer): next layer of the neuron

        Returns:
            None
        '''
        for i in range(len(self.neurons)):
            error = self.neurons[i]._backpropagation(output_error, next_layer, i)
            self.neurons[i].error = error
    
    def _update_weights(self, learning_rate, x):
        '''
        Update the weights of the layer

        Parameters:
            learning_rate (float): learning rate of the network
            x (numpy array): input to the layer

        Returns:
            None
        '''
        for neuron in self.neurons:
            neuron.weights -= learning_rate * neuron.error * x
    
    def _update_bias(self , learning_rate):
        '''
        Update the bias of the layer

        Parameters:
            learning_rate (float): learning rate of the network

        Returns:
            None
        '''
        for neuron in self.neurons:
            neuron.bias -= learning_rate * neuron.error[0]
    
    def __repr__(self):
        return 'Layer with %d neurons' % len(self.neurons)
    
class InputLayer(Layer):
    '''
    Input Layer class

    Attributes:
        neurons (list): list of neurons in the layer
        input_size (int): input size of the layer
        output_size (int): output size of the layer
        activation (function): activation function of the layer
        output (numpy array): output of the layer

    Methods:
        _feed_forward: feed forward the input to the layer
    '''
    def __init__(self, input_size):
        '''
        Constructor for Input Layer class

        Parameters:
            input_size (int): input size of the layer

        Returns:
            None
        '''
        super().__init__(input_size, input_size)
    
    def feed_forward(self, inputs):
        '''
        Feed forward the input to the layer

        Parameters:
            inputs (numpy array): input to the layer

        Returns:
            inputs (numpy array): outputs of the layer
        '''
        return inputs
    
    def __repr__(self):
        return 'Input Layer with %d neurons' % len(self.neurons)
    
    
class NeuralNetwork:
    '''
    Neural Network class

    Attributes:
        layers (list): list of layers in the network
        input_layer (InputLayer): input layer of the network
        output_layer (Layer): output layer of the network

    Methods:
        _init_layers: initialize the layers of the network
        _forward: feed forward the input to the network
        _backward: backpropagate the error to the previous layer
        _mse: calculate the mean squared error
        _mse_derivative: calculate the derivative of the mean squared error
        fit: fit the network to the data
        predict: predict the output of the network
        calculate_accuracy: calculate the accuracy of the network
        evaluate: evaluate the network on the data
    '''
    def __init__(self, layers):
        '''
        Constructor for Neural Network class

        Parameters:
            layers (list): list of layers in the network

        Returns:
            None
        '''
        layers = self._init_layers(layers)
        self.layers = layers[1:-1]
        self.input_layer = layers[0]
        self.output_layer = layers[-1]
    
    def _init_layers(self,layers):
        '''
        Initialize the layers of the network

        Parameters:
            layers (list): list of layers in the network

        Returns:
            layers (list): list of layers in the network
        '''
        total_neurons = 0
        for layer in layers:
            total_neurons += layer.input_size * layer.output_size

        for i in range(len(layers)-1):
            for _ in range(layers[i].output_size):
                rng = np.random.default_rng()
                weights = rng.uniform(-1/total_neurons, 1/total_neurons ,layers[i].input_size)
                layers[i].neurons.append(Neuron(weights, layers[i].activation))

        for _ in range(layers[-1].output_size):
            weights = np.ones(layers[-1].input_size)
            layers[-1].neurons.append(Neuron(weights, layers[-1].activation))
        return layers

    def _forward(self, input):
        '''
        Feed forward the input to the network

        Parameters:
            input (numpy array): input to the network

        Returns:
            output (numpy array): output of the network
        '''
        output = input
        for layer in self.layers:
            output = layer._feed_forward(output)
            layer.output = output
        output = self.output_layer._feed_forward(output)
        return output
    
    
    def _backward(self, output_error):
        '''
        Backpropagate the error to the previous layer

        Parameters:
            output_error (float): error of the output layer

        Returns:
            None
        '''
        next_layer = self.output_layer
        for layer in reversed(self.layers):
            layer._backpropagation(output_error, next_layer)
            next_layer = layer

    def _mse(self, y_true, y_pred):
        '''
        Calculate the mean squared error

        Parameters:
            y_true (numpy array): true output
            y_pred (numpy array): predicted output

        Returns:
            error (float): mean squared error
        '''
        return np.square(np.subtract(y_true, y_pred)).mean()

    def _mse_derivative(self, y_true, y_pred):
        '''
        Calculate the derivative of the mean squared error

        Parameters:
            y_true (numpy array): true output
            y_pred (numpy array): predicted output

        Returns:
            error (float): derivative of the mean squared error
        '''
        return (y_pred - y_true) * y_pred * (1 - y_pred)
    
    def fit(self, input, target, epochs, learning_rate):
        '''
        Fit the network to the data

        Parameters:
            input (numpy array): input to the network
            target (numpy array): target output
            epochs (int): number of epochs
            learning_rate (float): learning rate of the network

        Returns:
            None
        '''
        for epoch in range(epochs):
            error = 0
            for x, y in zip(input, target):
                self.input_layer.output = x
                output = self._forward(x)
                error += self._mse(y, output)
                output_error = self._mse_derivative(y, output)if (self.output_layer.activation == sigmoid) else (output-y)
                self.output_layer.neurons[0].error = output_error[0]
                self._backward(output_error)
                a = x
                for layer in self.layers:
                    layer._update_weights(learning_rate, a)
                    layer._update_bias(learning_rate)
                    a = layer.output
            error /= len(input)
            print('Epoch: %d/%d, Error: %f' % (epoch+1, epochs, error))
    
    def predict(self, input):
        '''
        Predict the output of the network

        Parameters:
            input (numpy array): input to the network

        Returns:
            output (numpy array): output of the network
        '''
        output = []
        for x in input:
            output.append(self._forward(x)[0])
        return output
    
    def calculate_accuracy(self, y_pred, y_test):
        '''
        Calculate the accuracy of the network

        Parameters:
            y_pred (numpy array): predicted output
            y_test (numpy array): true output

        Returns:
            error (float): accuracy of the network
        '''
        return self._mse(y_test, y_pred)
    
    def evaluate(self, input, target):
        '''
        Evaluate the network on the data

        Parameters:
            input (numpy array): input to the network
            target (numpy array): target output

        Returns:
            None
        '''
        output = self.predict(input)
        error = self.calculate_accuracy(output, target)
        print('Error: %f' % error)

    def __repr__(self):
        return 'Neural Network with %d layers' % len(self.layers)
    
    def __str__(self):
        return 'Neural Network with %d layers' % len(self.layers)


def sigmoid(x):
    '''
    Sigmoid activation function

    Parameters:
        x (float): input to the activation function

    Returns:
        x (float): output of the activation function
    '''
    return 1/(1 + np.exp(-x))

def relu(x):
    '''
    ReLU activation function

    Parameters:
        x (float): input to the activation function

    Returns:
        x (float): output of the activation function
    '''
    return max(0 , x)

def linear(x):
    '''
    Linear activation function

    Parameters:
        x (float): input to the activation function

    Returns:
        x (float): output of the activation function
    '''
    return x