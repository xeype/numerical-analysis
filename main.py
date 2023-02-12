import numpy as np
import matplotlib.pyplot as plt


# function
def f(x):
    return np.arctan(x) - x


# number of nodes
N = 3
# N = 5
# N = 15
# N = 20
# N = 25

# Interval and nodes
a, b = -2, 2
x = np.linspace(a, b, N)
y = f(x)


def newton_interpolation(x, y, N, x_eval):
    F = np.zeros((N, N))
    F[:, 0] = y

    for j in range(1, N):
        for i in range(N - j):
            F[i][j] = (F[i + 1][j - 1] - F[i][j - 1]) / (x[i + j] - x[i])

    p = 0
    for i in range(N):
        term = F[0][i]
        for j in range(i):
            term *= (x_eval - x[j])
        p += term

    return p


def lagrange_interpolation(x, y, N, x_eval):
    p = 0
    for i in range(N):
        term = y[i]
        for j in range(N):
            if j != i:
                term *= (x_eval - x[j]) / (x[i] - x[j])
        p += term

    return p


def chebyshev_interpolation(x_nodes, y_nodes, x_eval):
    N = len(x_nodes)
    c = np.zeros(N)
    for i in range(N):
        c[i] = y_nodes[i]
        for j in range(N):
            if j != i:
                c[i] *= (x_eval - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
    return c.sum()


# Chebyshev nodes
x_chebyshev = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (2 * np.arange(1, N + 1) - 1) / (2 * N))
y_chebyshev = f(x_chebyshev)

# evaluate the Newton, Lagrange, and Chebyshev Interpolations
x_eval = np.linspace(a, b, 1000)
y_eval = f(x_eval)
y_newton = np.array([newton_interpolation(x, y, N, xi) for xi in x_eval])
y_lagrange = np.array([lagrange_interpolation(x, y, N, xi) for xi in x_eval])
y_chebyshev = np.array([chebyshev_interpolation(x_chebyshev, y_chebyshev, xi) for xi in x_eval])

# build graphs
plt.plot(x_eval, y_eval, 'r', label='True Function')
plt.plot(x_eval, y_newton, 'b', label='Newton Interpolation')
plt.plot(x_eval, y_lagrange, 'g', label='Lagrange Interpolation')
plt.plot(x_eval, y_chebyshev, 'y', label='Chebyshev Interpolation')
plt.legend()
plt.show()
