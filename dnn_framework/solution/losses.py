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

        # loss = -np.sum(target_onehot * np.log(y))

        # CE grad
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
    L1 = np.linalg.norm(np.exp(x), 1, axis=1)
    y = np.exp(x) / np.expand_dims(L1, axis=1)

    return y

def softmax_grad(y, output_grad):

    # y: [2,3]   :  [ [1.0, 2.0, 3.0], [2.0, 1.0, 5.0] ]
    # D : [2,3,3]
    I = y.shape[1]
    J = y.shape[1]
    D = np.empty([y.shape[0], I, J], 'float')
    for b in range(len(y)):
        for i in range(I):
            for j in range(J):
                if i != j:
                    D[b][i][j] = -y[b][i] * y[b][j]
                elif i == j:
                    D[b][i][j] = y[b][j] * (1 - y[b][j])


    input_grad = np.zeros(output_grad.shape)
    for b in range(len(D)):
        for i in range(I):
            res = D[b][i] * output_grad[b]
            res_sum = np.sum(res)
            input_grad[b][i] = res_sum

    # input_grad2 = np.sum(D, axis=2) * output_grad

    return input_grad
