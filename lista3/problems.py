import numpy as np

def griewank(P):
    a = (P**2).sum(axis=1) / 4000
    b = np.cos(P / np.sqrt(np.arange(1, P.shape[1] + 1))).prod(axis=1)
    return -(a - b + 1)

def griewank_init(d, N):
    return np.random.uniform(-600, 600, size=(N, d))

def rastrigin(P):
    A = 10
    a = A * P.shape[1]
    b = (P**2 - A * np.cos(2 * np.pi * P)).sum(axis=1)
    return -(a + b)

def rastrigin_init(d, N):
    return np.random.uniform(-5.12, 5.12, size=(N, d))

def schwefel(P):
    a = 418.9829 * P.shape[1]
    b = (P * np.sin(np.sqrt(np.abs(P)))).sum(axis=1)
    return -(a - b)

def schwefel_init(d, N):
    return np.random.uniform(-500, 500, size=(N, d))

def ackeley(P):
    A, B, C = 20, .2, 2 * np.pi
    a = A * np.exp(-B * np.sqrt((P**2).mean(axis=1)))
    b = np.exp(np.cos(C * P).mean(axis=1))
    return -(-a - b + A + np.exp(1))

def ackeley_init(d, N):
    return np.random.uniform(-15, 30, size=(N, d))

def sphere(P):
    return -(P**2).sum(axis=1)

def sphere_init(d, N):
    return np.random.uniform(-5.12, 5.12, size=(N, d))