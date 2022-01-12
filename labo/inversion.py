import numpy as np
import matplotlib.pyplot as plt

SIZE = 4
mu = 0.01  # 0.001
# A = np.array([[3, 4, 1],
#               [5, 2, 3],
#               [6, 2, 2]])
# A = np.array([[3, 4, 1, 2, 1, 5],
#               [5, 2, 3, 2, 2, 1],
#               [6, 2, 2, 6, 4, 5],
#               [1, 2, 1, 3, 1, 2],
#               [1, 5, 2, 3, 3, 3],
#               [1, 2, 2, 4, 2, 1]])
A = np.array([[2, 1, 1, 2],
              [1, 2, 3, 2],
              [2, 1, 1, 2],
              [3, 1, 4, 1]])
I = np.identity(SIZE)
B = np.zeros((SIZE, SIZE))


def loss():
    return np.sum(np.square((np.matmul(B, A) - I)))


def grad():
    return np.matmul(2*(np.matmul(B, A) - I), np.transpose(A))


def step():
    back_prop = -mu * grad()
    return B + back_prop


epochs = 1000
l = [None]*epochs

for i in range(epochs):
    l[i] = loss()
    B = step()

print("I:", I)
print("B:", B)
print("A:", A)
print("Res:", np.matmul(B, A))
print("Loss:", l[epochs-1])
# print(l)

x = range(epochs)
plt.plot(x, l)
plt.show()
