from dnn_framework.loss import Loss
import numpy as np

class CrossEntropyLoss(Loss):
    """
    This is the base class of every loss function.
    Every loss class must inherit from this class and must overload the calculate method.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor
        :param target: The target tensor
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """

        # Somme sur les classes et moyenne sur les exemples

        # Convert target to one-hot
        target_onehot = np.zeros(x.shape)
        target = np.expand_dims(target, axis=1)
        np.put_along_axis(target_onehot, target, 1, axis=1)

        y = softmax(x)

        ln = np.log(y)
        mul = target_onehot * ln
        sum = np.sum(mul) / x.shape[0]
        loss = -1 * sum

        loss2 = -np.sum(target_onehot * np.log(y))

        # BCE grad
        output_grad = -1 * target_onehot / y

        # Softmax grad
        input_grad = softmax_grad(y, output_grad)


        return loss, input_grad




class MeanSquaredErrorLoss(Loss):
    """
    This is the base class of every loss function.
    Every loss class must inherit from this class and must overload the calculate method.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor
        :param target: The target tensor
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """

        n_elements = x.shape[0]*x.shape[1]

        loss = np.mean(np.square(x - target))
        input_grad = 2/n_elements * (x - target)

        return loss, input_grad


def softmax(x):
    # TODO: Voir eq.72

    sum_exp = np.sum(np.exp(x), axis=0)
    y_0 = np.exp(x) / sum_exp

    e_x = np.exp(x - np.max(x))
    y2 = e_x / np.sum(e_x, axis=0)

    L1 = np.linalg.norm(np.exp(x), 1, axis=1)
    y = np.exp(x) / np.expand_dims(L1, axis=1)

    return y

def softmax_grad(input, input_grad):
    pass