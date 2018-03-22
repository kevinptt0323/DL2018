#!/usr/bin/env python3
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_d(x):
    return x * (1.0 - x)

# 3 layer NN
def train(X, Y, epochs=1000, learning_rate=0.001, bias=True):
    hidden_layer_size = 2
    if bias:
        X = np.pad(X, ((0, 0), (1, 0)), 'constant', constant_values=(1,))

    Y = np.array([Y]).T

    global W1, W2, L2, L3

    np.random.seed(8787)
    W1 = np.random.uniform(-1, 1, (X.shape[1], hidden_layer_size))
    W2 = np.random.uniform(-1, 1, (hidden_layer_size+bias, 1))

    def forward(X):
        global L2, L3, E
        L2 = sigmoid(np.dot(X, W1))
        if bias:
            L2 = np.pad(L2, ((0, 0), (1, 0)), 'constant', constant_values=(1,))
        L3 = sigmoid(np.dot(L2, W2))

    def backward(X, E):
        global W1, W2
        grad_L3 = E * sigmoid_d(L3)
        grad_L2 = np.dot(grad_L3, W2.T) * sigmoid_d(L2)
        grad_L2 = grad_L2[:,bias:]
        W2 += np.dot(L2.T, grad_L3) * learning_rate
        W1 += np.dot(X.T, grad_L2) * learning_rate

    for i in range(epochs):
        forward(X)
        E = Y - L3
        backward(X, E)
        if i % 10000 == 0:
            print('%6d' % i, L3.T[0])

    for data in X:
        forward(np.array([data]))
        print(data[bias:], ' -> ', L3[0])

# 4 layer NN
def train_2(X, Y, epochs=1000, learning_rate=0.001, bias=True):
    hidden_layer_size = 2
    if bias:
        X = np.pad(X, ((0, 0), (1, 0)), 'constant', constant_values=(1,))

    Y = np.array([Y]).T

    global W1, W2, W3, L2, L3, L4

    np.random.seed(1234)
    W1 = np.random.uniform(-1, 1, (X.shape[1], hidden_layer_size))
    W2 = np.random.uniform(-1, 1, (hidden_layer_size+bias, hidden_layer_size))
    W3 = np.random.uniform(-1, 1, (hidden_layer_size+bias, 1))

    def forward(X):
        global L2, L3, L4
        L2 = sigmoid(np.dot(X, W1))
        if bias:
            L2 = np.pad(L2, ((0, 0), (1, 0)), 'constant', constant_values=(1,))
        L3 = sigmoid(np.dot(L2, W2))
        if bias:
            L3 = np.pad(L3, ((0, 0), (1, 0)), 'constant', constant_values=(1,))
        L4 = sigmoid(np.dot(L3, W3))

    def backward(X, E):
        global W1, W2, W3, W4
        grad_L4 = E * sigmoid_d(L4)
        grad_L3 = np.dot(grad_L4, W3.T) * sigmoid_d(L3)
        grad_L3 = grad_L3[:,bias:]
        grad_L2 = np.dot(grad_L3, W2.T) * sigmoid_d(L2)
        grad_L2 = grad_L2[:,bias:]
        W3 += np.dot(L3.T, grad_L4) * learning_rate
        W2 += np.dot(L2.T, grad_L3) * learning_rate
        W1 += np.dot(X.T, grad_L2) * learning_rate

    for i in range(epochs):
        forward(X)
        E = Y - L4
        backward(X, E)
        if i % 10000 == 0:
            print('%6d' % i, L4.T[0])

    for data in X:
        forward(np.array([data]))
        print(data[bias:], ' -> ', L4[0])


def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    Y = np.array([0, 1, 1, 0])

    train(X, Y,
          epochs=100000,
          learning_rate=0.1,
          bias=True)

if __name__ == '__main__':
    main()
