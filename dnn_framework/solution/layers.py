import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, intput_size, output_size):
        self.w = np.zeros([intput_size, output_size])
        self.b = np.zeros([1, self.output_size])

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        parameters = {"w": self.w, "b": self.b}
        return parameters

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """

        y = self.w * x + self.b

        cache = None  # TODO
        return (y, cache)

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        raise NotImplementedError()


class BatchNormalization(Layer):

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        raise NotImplementedError()

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        # TODO: Lissage de mu et sigma avec alpha & ajustement de y avec gamma et beta (eq. 66)
        mu = np.mean(x)
        sigma = np.std(x)

        y = (x - mu)/sigma

        cache = None  # TODO
        return (y, cache)

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
        raise NotImplementedError()


class ReLU(Layer):

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        raise NotImplementedError()

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """
        y = x.copy()
        y[y < 0] = 0

        cache = None  # TODO
        return (y, cache)

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        raise NotImplementedError()


class Sigmoid(Layer):

    def get_parameters(self):
        """
        This method returns the parameters of the layer.
        The parameters are learned by the optimizer.
        :return: A dictionary (str: np.array) of the parameters
        """
        raise NotImplementedError()

    def get_buffers(self):
        """
        This method returns the buffers of the layer.
        The buffers are not learned by the optimizer.
        :return: A dictionary (str: np.array) of the buffers
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward pass of the layer.
        :param x: The input tensor
        :return: A tuple containing the output value and the cache (y, cache)
        """

        y = 1 / (1 + np.exp(-x))

        cache = None  # TODO
        return (y, cache)

    def backward(self, output_grad, cache):
        """
        This method performs the backward pass.
        :param output_grad: The gradient with respect to the output
        :param cache: The cache returned by the forward method
        :return: A tuple containing the gradient with respect to the input and
                 a dictionary containing the gradient with respect to each parameter indexed with the same key
                 as the get_parameters() dictionary.
        """
        raise NotImplementedError()