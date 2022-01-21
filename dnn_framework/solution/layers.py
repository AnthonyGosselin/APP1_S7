import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        # self.w = np.zeros([output_size, intput_size])
        # self.b = np.zeros([output_size, 1])

        # Initialize the weights with normal distribution
        self.w = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / (input_size + output_size)),
                                  size=(output_size, input_size))
        self.b = np.random.normal(loc=0.0,
                                  scale=np.sqrt(2 / output_size),
                                  size=(output_size,))

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        parameters = {"w": self.w, "b": self.b}
        return parameters

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        B = self.b[np.newaxis]  # Turns one axis to 2 axis for transpose
        y = x @ self.w.T + B
        cache = {"input": x}
        return y, cache

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        X = cache['input']

        input_grad = output_grad.T @ self.w

        w_grad = output_grad @ X
        b_grad = np.sum(output_grad, axis=1)            # TODO: CHECK IF SUM IS ON CORRECT AXIS

        grad_dict = {"w": w_grad, "b": b_grad}

        return input_grad, grad_dict


class BatchNormalization(Layer):

    def __init__(self, size, g_mean=None, g_variance=None, alpha=0.5):
        super().__init__()
        self.gamma = np.ones(size)
        self.beta = np.zeros(size)

        self.global_mean = np.zeros(size)
        if g_mean:
            self.global_mean = g_mean

        self.global_variance = np.zeros(size)
        if g_variance:
            self.global_variance = g_variance

        self.alpha = alpha

        self.EPSILON = 1e-20

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        parameters = {'gamma': self.gamma, 'beta': self.beta}
        return parameters

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        buffer_dict = {'global_mean': self.global_mean, 'global_variance': self.global_variance}
        return buffer_dict

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """

        # Lissage de mu et sigma avec alpha & ajustement de y avec gamma et beta (eq. 66)
        if self.is_training():
            mu = np.sum(x, axis=0) / x.shape[0]
            self.global_mean = (1 - self.alpha) * self.global_mean + self.alpha * mu
            sigma2 = np.sum(np.square(x-mu), axis=0) / x.shape[0]
            self.global_variance = (1 - self.alpha) * self.global_variance + self.alpha * sigma2
        else:
            mu = self.global_mean
            sigma2 = self.global_variance

        x_hat = (x - mu) / np.sqrt(sigma2 + self.EPSILON)

        y = self.gamma * x_hat + self.beta

        cache = {'input': x, "x_hat": x_hat, 'y': y, 'mu': mu, 'sigma2': sigma2}
        return y, cache

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        # TODO: Tuple: (eq70, {eq71}

        x = cache["input"]
        sigma2 = cache["sigma2"]
        mu = cache["mu"]
        x_hat = cache["x_hat"]
        sigma_epsilon = sigma2 + self.EPSILON
        sigma_epsilon_root = np.sqrt(sigma_epsilon)
        M = len(x)

        # TODO: CHECK IF SUMS ARE OK

        dldy = output_grad

        # EQ. 67
        dldx_hat = dldy * self.gamma

        # EQ. 68
        dldsigma2 = np.sum( dldx_hat * (x - mu) * (-0.5) * np.power( sigma_epsilon, -3/2 ), axis=0 )

        # EQ. 69
        dldmu = (-1 * np.sum(dldx_hat, axis=0) / sigma_epsilon_root) - 2/M * dldsigma2 * np.sum(x-mu, axis=0)

        # EQ. 70
        dldx = dldx_hat / sigma_epsilon_root + 2/M * dldsigma2 * (x - mu) + 1/M * dldmu

        # EQ. 71
        dldgamma = np.sum(dldy * x_hat, axis=0)
        dldbeta = np.sum(dldy, axis=0)

        grad_dict = {'gamma': dldgamma, 'beta': dldbeta}

        return dldx, grad_dict


class ReLU(Layer):

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        y = x.copy()
        y[y < 0] = 0

        cache = {"input": x}  # TODO
        return y, cache

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        # dldx = dydx * dldy
        X = cache['input']
        Y = X.copy()
        Y[Y < 0] = 0
        Y[Y > 0] = 1
        x_grad = Y * output_grad  # [N,I] = [N,I] * [N,I]

        grad_dict = {}  #TODO: ???

        return x_grad, grad_dict


class Sigmoid(Layer):

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """

        y = 1 / (1 + np.exp(-x))

        cache = {'input': x}
        return y, cache

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """

        X = cache['input']

        Y = 1 / (1 + np.exp(-X))
        dydx = (1 - Y) * Y
        x_grad = dydx * output_grad

        grad_dict = {}  # TODO: ???

        return x_grad, grad_dict
