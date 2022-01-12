import numpy as np
import matplotlib.pyplot as plt

deg = 9
N = 1 + deg
mu = 0.05
a = np.zeros([N])

xi = np.array([-0.95, -0.82, -0.62, -0.43, -0.17, -0.07, 0.25, 0.38, 0.61, 0.79, 1.04])
y = np.array([0.02, 0.03, -0.17, -0.12, -0.37, -0.25, -0.10, 0.14, 0.53, 0.71, 1.53])

# Create x from length N
def x_vec(x_i):
    x = [None] * N
    powers = range(N)
    for i in powers:
        x[i] = np.power(x_i, i)
    return np.transpose(x)

x = x_vec(xi)

def loss(y_chap, y):
    l = np.sum(np.square(y_chap-y))
    return l

def grad(y_chap):
    y_y = y_chap-y
    y_yx = y_y * np.transpose(x)
    sum = 2 * np.sum(y_yx, axis=1)
    return sum

def step(v, y_chap):
    # back_prop = 0
    # for i in range(len(xi)):
    back_prop = (-mu * grad(y_chap))
    return v + back_prop


epochs = 10000
l = [None] * epochs
for i in range(epochs):
    y_chap = np.matmul(a, np.transpose(x))
    l[i] = loss(y_chap, y)
    a = step(a, y_chap)


print(a)
for yi, y_chapi in zip(y, y_chap):
    print(yi, " ", y_chapi)
# print(y_chap)
print("Loss:", l[epochs-1])

x_range = range(epochs)
plt.plot(x_range, l)
plt.show()

plt.figure()
x_range = np.linspace(-1.25, 1.25, 100)
y_range = np.matmul(a, np.transpose(x_vec(x_range)))
plt.plot(x_range, y_range)
plt.scatter(xi, y)
plt.show()