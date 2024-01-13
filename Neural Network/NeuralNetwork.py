import numpy as np

class Neuron:
    '''
    Neuron class

    Parameters:
    -----------
    weights: numpy array
        The weights of the neuron

    activation: function
        The activation function of the neuron

    Attributes:
    -----------
    bias: float
        The bias of the neuron

    error: float
        The error of the neuron

    output: float
        The output of the neuron

    Methods:
    --------
    _feed_forward(inputs)
        This function feeds forward the input to the neuron.

    _backpropagation(output_error, next_layer, idx)
        This function backpropagates the error to the previous layer.
    '''
    def __init__(self, weights, activation):
        '''
        Constructor for Neuron class

        Parameters:
        ----------
        
        weights: numpy array
            The weights of the neuron

        activation: function
            The activation function of the neuron

        Returns:
        --------
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
        ----------
        inputs: numpy array
            The input to the neuron

        Returns:
        --------
        output: float
            The output of the neuron
        '''
        total = np.dot(self.weights, inputs) + self.bias
        self.output = self.activation(total) if self.activation else total
        return self.output
    
    def _backpropagation(self, next_layer, idx):
        '''
        Backpropagate the error to the previous layer

        Parameters:
        ----------
        next_layer: Layer
            The next layer of the neuron

        idx: int
            The index of the neuron in the next layer

        Returns:
        --------
        error: float
            The error of the neuron
        '''
        err = 0
        for neuron in next_layer.neurons:
            err += neuron.error*neuron.weights[idx]
        self.error = err * self.output * (1 - self.output)
        return self.error

    
class Layer:
    '''
    Layer class

    Attributes:
    ----------
    neurons: list
        List of neurons in the layer

    input_size: int
        Input size of the layer

    output_size: int
        Output size of the layer

    activation: function
        Activation function of the layer

    output: numpy array
        Output of the layer

    Methods:
    --------
    _feed_forward(inputs)
        This function feeds forward the input to the layer.

    _backpropagation(output_error, next_layer)
        This function backpropagates the error to the previous layer.

    _update_weights(learning_rate, x)
        This function updates the weights of the layer.

    _update_bias(learning_rate)
        This function updates the bias of the layer.
    '''
    def __init__(self, output_size, activation=None):
        '''
        Constructor for Layer class

        Parameters:
        ----------
        output_size: int
            Output size of the layer

        activation: function
            Activation function of the layer

        Returns:
        --------
        None
        '''
        self.activation = activation if activation else linear
        self.neurons = []
        self.input_size = 0
        self.output_size = output_size

    def _feed_forward(self, inputs):
        '''
        Feed forward the input to the layer

        Parameters:
        ----------
        inputs: numpy array
            The input to the layer

        Returns:
        --------
        outputs: numpy array
            The outputs of the layer
        '''
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron._feed_forward(inputs))
        self.output = np.array(outputs)
        return self.output
    
    def _backpropagation(self, next_layer):
        '''
        Backpropagate the error to the previous layer

        Parameters:
        ----------
        next_layer: Layer
            The next layer of the neuron

        Returns:
        --------
        None
        '''
        for i in range(len(self.neurons)):
            self.neurons[i]._backpropagation(next_layer, i)
    
    def _update_weights(self, learning_rate, x):
        '''
        Update the weights of the layer

        Parameters:
        ----------
        learning_rate: float
            Learning rate of the network

        x: numpy array
            Input to the layer

        Returns:
        --------
        None
        '''
        for neuron in self.neurons:
            neuron.weights -= learning_rate * neuron.error * x
    
    def _update_bias(self , learning_rate):
        '''
        Update the bias of the layer

        Parameters:
        ----------
        learning_rate: float
            Learning rate of the network

        Returns:
        --------
        None
        '''
        for neuron in self.neurons:
            neuron.bias -= learning_rate * neuron.error
    
    def __str__(self):
        return f'Layer ({self.input_size}, {self.output_size}) with %d neurons' % len(self.neurons)
    
    def __repr__(self):
        return self.__str__()
    
class InputLayer(Layer):
    '''
    Input Layer class

    Parameters:
    ----------
    input_size: int
        Input size of the layer

    Attributes:
    ----------
    neurons: list
        List of neurons in the layer

    output_size: int
        Output size of the layer

    activation: function
        Activation function of the layer

    output: numpy array
        Output of the layer

    Methods:
    --------
    _feed_forward(inputs)
        This function feeds forward the input to the layer.

    _backpropagation(output_error, next_layer)
        This function backpropagates the error to the previous layer.

    _update_weights(learning_rate, x)
        This function updates the weights of the layer.

    _update_bias(learning_rate)
        This function updates the bias of the layer.
    '''
    def __init__(self, input_size):
        '''
        Constructor for Input Layer class

        Parameters:
        ----------
        input_size: int
            Input size of the layer

        Returns:
        --------
        None
        '''
        super().__init__(input_size, input_size)
    
    def _feed_forward(self, inputs):
        '''
        Feed forward the input to the layer

        Parameters:
        ----------
        inputs: numpy array
            The input to the layer

        Returns:
        --------
        inputs: numpy array
            The input to the layer
        '''
        return inputs
    
    def __str__(self):
        return f'Input Layer ({self.input_size}, {self.output_size}) with %d neurons' % len(self.neurons)
    
    def __repr__(self):
        return self.__str__()
    
    
class NeuralNetwork:
    '''
    Neural Network class

    Parameters:
    ----------
    layers: list
        List of layers in the network

    Attributes:
    ----------
    layers: list
        List of layers in the network

    input_layer: InputLayer
        Input layer of the network

    output_layer: Layer
        Output layer of the network

    Methods:
    --------
    fit(input, target, epochs, learning_rate)
        fit the network to the data

    predict(input)
        predict the output of the network

    calculate_accuracy(y_pred, target)
        calculate the accuracy of the network

    evaluate(input, target)
        evaluate the network on the data

    summary()
        print the summary of the network
    '''
    def __init__(self, layers):
        '''
        Constructor for Neural Network class

        Parameters:
        ----------
        layers: list
            List of layers in the network

        Returns:
        --------
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
        ----------
        layers: list
            List of layers in the network

        Returns:
        --------
        layers: list
        '''
        last_output_size = layers[0].output_size
        for layer in layers:
            layer.input_size = last_output_size
            last_output_size = layer.output_size

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
        ----------
        input: numpy array
            The input to the network

        Returns:
        --------
        output: numpy array
            The output of the network
        '''
        output = input
        self.input_layer.output = output
        for layer in self.layers:
            output = layer._feed_forward(output)
            layer.output = output
        output = self.output_layer._feed_forward(output)
        return output
    
    
    def _backward(self, y_true):
        '''
        Backpropagate the error to the previous layer

        Parameters:
        ----------
        y_true: numpy array
            The target output

        Returns:
        --------
        None
        '''
        next_layer = self.output_layer
        output = self.output_layer.output

        # backpropagate the output layer
        for i,neuron in enumerate(self.output_layer.neurons):
            neuron.error = _mse_derivative(y_true[i], output[i]) * (_sigmoid_derivative(output[i]) if (self.output_layer.activation == sigmoid) else 1)

        # backpropagate the hidden layers
        for layer in reversed(self.layers):
            layer._backpropagation(next_layer)
            next_layer = layer
    
    def _update_weights(self, learning_rate, x):
        '''
        Update the weights of the network

        Parameters:
        ----------
        learning_rate: float
            Learning rate of the network

        x: numpy array
            Input to the network

        Returns:
        --------
        None
        '''
        a = x
        for layer in self.layers:
            layer._update_weights(learning_rate, a)
            layer._update_bias(learning_rate)
            a = layer.output
        self.output_layer._update_weights(learning_rate, a)
        self.output_layer._update_bias(learning_rate)

    def _mse(self, y_true, y_pred):
        '''
        Calculate the mean squared error

        Parameters:
        ----------
        y_true : numpy array
            true output

        y_pred : numpy array
            predicted output

        Returns:
        --------
        error : float
            mean squared error
        '''
        return np.square(np.subtract(y_true, y_pred)).mean()
    
    def fit(self, input, target, epochs, learning_rate):
        '''
        Fit the network to the data

        Parameters:
        ----------
        input : 2d numpy array
            The input to the network

        target : 2d numpy array
            The target output

        epochs : int
            Number of epochs to train the network

        learning_rate : float
            Learning rate of the network

        Returns:
        --------
        None
        '''
        if(target.shape[1] != self.output_layer.output_size):
            print('Error: Output size of the network does not match the target output size')
            return
        for epoch in range(epochs):
            error = 0
            for x, y in zip(input, target):
                output = self._forward(x)
                error += self._mse(y, output)
                self._backward(y)
                self._update_weights(learning_rate, x)
            error /= len(input)
            print('Epoch: %d/%d, Error: %f' % (epoch+1, epochs, error))
    
    def predict(self, input):
        '''
        Predict the output of the network

        Parameters:
        ----------
        input : numpy array
            The input to the network

        Returns:
        --------
        output : numpy array
        '''
        output = []
        for x in input:
            output.append(self._forward(x)[0])
        return output
    
    def calculate_accuracy(self, y_pred, y_test):
        '''
        Calculate the accuracy of the network

        Parameters:
        ----------
        y_pred : numpy array
            The predicted output

        y_test : numpy array
            The target output

        Returns:
        --------
        error : float
            The mean squared error
        '''
        return self._mse(y_test, y_pred)
    
    def evaluate(self, input, target):
        '''
        Evaluate the network on the data

        Parameters:
        ----------
        input : numpy array
            The input to the network
            
        target : numpy array
            The target output

        Returns:
        --------
        None
        '''
        output = self.predict(input)
        error = self.calculate_accuracy(output, target)
        print('Error: %f' % error)
    
    def summary(self):
        '''
        Print the summary of the network

        Parameters:
        ----------
        None

        Returns:
        --------
        None
        '''
        print(self)
        print('[')
        print(self.input_layer)
        for layer in self.layers:
            print(layer)
        print(self.output_layer)
        print(']')

    def __str__(self):
        return 'Neural Network with %d layers' % (len(self.layers)+2)
    
    def __repr__(self):
        return self.__str__()

def _mse_derivative(y_true, y_pred):
    '''
    Calculate the derivative of the mean squared error

    Parameters:
    ----------
    y_true : numpy array
        true output

    y_pred : numpy array
        predicted output

    Returns:
    --------
    error : float
        derivative of the mean squared error
    '''
    return (y_pred - y_true)

def _sigmoid_derivative(x):
    '''
    Calculate the derivative of the sigmoid function

    Parameters:
    ----------
    x : float
        input to the sigmoid function

    Returns:
    --------
    x : float
        derivative of the sigmoid function
    '''
    return x * (1 - x)


def sigmoid(x):
    '''
    Sigmoid activation function

    Parameters:
    ----------
    x : float
        input to the activation function

    Returns:
    --------
    x : float
        output of the activation function
    '''
    return 1/(1 + np.exp(-x))

def relu(x):
    '''
    ReLU activation function

    Parameters:
    ----------
    x : float
        input to the activation function

    Returns:
    --------
    x : float
        output of the activation function
    '''
    return max(0 , x)

def linear(x):
    '''
    Linear activation function

    Parameters:
    ----------
    x : float
        input to the activation function

    Returns:
    --------
    x : float
        output of the activation function
    '''
    return x