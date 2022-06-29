# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import sys
import torch
import time

import torchvision.datasets

import operation as common
from input_data import mnist
# import linalg as common
# import numpy as np
# import tensorflow as tf
# from keras.datasets import reuters, mnist, imdb

def SimpleRNN(input, state, u, w, bias, v, c):
    ux = torch.matmul(u, input)
    ws = torch.matmul(w, state)
    a = torch.add(ux, ws)
    a = torch.add(a, bias)
    s_next = torch.tanh(a)
    output = torch.matmul(v, s_next) + c
    return a, s_next, output

def inference(input, label, state, u, w, bias, v, c):
    num = label.shape[1]
    ux = torch.matmul(u, input)
    ws = torch.matmul(w, state)
    a = torch.add(ux, ws)
    a = torch.add(a, bias)
    s_next = torch.tanh(a)
    output = torch.matmul(v, s_next) + c
    prediction = common.softmax(output)
    loss = common.cross_entropy_with_softmax(label, output) / num
    print(torch.argmax(prediction, axis=0))
    print(torch.argmax(y_train, axis=0))
    print(torch.equal(torch.argmax(prediction, axis=0), torch.argmax(y_train, axis=0)))
    acc = torch.sum(torch.eq(torch.argmax(prediction, axis=0), torch.argmax(y_train, axis=0))) / num * 100
    return loss, acc

def train_init():
    init_u = torch.randn(hidden_size, dimensions)*0.01
    init_w = torch.randn(hidden_size, hidden_size)*0.01
    init_state = torch.zeros((hidden_size, train_nums))
    init_b = torch.randn(hidden_size, train_nums)*0.01
    init_v = torch.randn(10, hidden_size)*0.01
    init_c = torch.randn(10, train_nums)*0.01
    return init_state, init_u, init_w, init_b, init_v, init_c

def test_init():
    init_u = torch.randn(hidden_size, dimensions)*0.01
    init_w = torch.randn(hidden_size, hidden_size)*0.01
    init_state = torch.zeros((hidden_size, train_nums))
    init_b = torch.randn(hidden_size, train_nums)*0.01
    init_v = torch.randn(10, hidden_size)*0.01
    init_c = torch.randn(10, test_nums)*0.01
    return init_state, init_u, init_w, init_b, init_v, init_c

num_words = 30000
maxlen = 50
test_split = 0.3
# (X_train, y_train), (X_test, y_test) = reuters.load_data(num_words = num_words)
data=torchvision.datasets.MNIST
print(data.train_data.fdel)
mnist = mnist()
print(mnist.train.xs)
(X_train, y_train), (X_test, y_test) = (mnist.train.xs, mnist.train.ys), (mnist.test.xs, mnist.test.ys)

X_train = torch.swapaxes(X_train, 0, 1)
y_train = torch.swapaxes(y_train, 0, 1)

train_size = X_train.size
test_size = X_test.size
train_nums = y_train.shape[1]
print("train nums:", train_nums)
test_nums = y_test.shape[1]
print("test nums:", test_nums)
dimensions = 784
hidden_size = 128

scale = 0.01
ITER = 1000
rho1 = 1
rho2 = 1
rho3 = 1
alpha = 1
tao = 1

train_init_s, train_init_u, train_init_w, train_init_b, train_init_v, train_init_c = train_init()
test_init_s, test_init_u, test_init_w, test_init_b, test_init_v, test_init_c = test_init()
# print(train_init_s, train_init_u, train_init_w, train_init_b)

## for train
u = train_init_u
w = train_init_w
b = train_init_b
s_old = train_init_s
v = train_init_v
c = train_init_c
a, st, o = SimpleRNN(X_train, train_init_s, train_init_u, train_init_w, train_init_b, train_init_v, train_init_c)

## for test
# u = test_init_u
# w = test_init_w
# b = test_init_b
# s_old = test_init_s
# v = test_init_v
# c = test_init_c
# a, st, o = SimpleRNN(X_train, test_init_s, test_init_u, test_init_w, test_init_b, test_init_v, test_init_c)


s_new = st

lambda1 = torch.zeros(a.shape)
print("lambda1:", lambda1.shape)
lambda2 = torch.zeros(st.shape)
print("lambda2:", lambda2.shape)
lambda3 = torch.zeros(o.shape)
print("lambda3:", lambda3.shape)

train_acc = torch.zeros(ITER)
train_loss = torch.zeros(ITER)

## for train
# loss, acc = inference(X_train, y_train, s_old, u, w, b, v, c)
# print("training acc:", acc)
# print("training loss:", loss)

## for test
# loss, acc = inference(X_test, y_test, s_old, u, w, b, v, c)
# loss_test = np.zeros(ITER)
# acc_test = np.zeros(ITER)
# print("testing acc:", acc)
# print("testing loss:", loss)
print("rho1:", rho1)
print("rho2:", rho2)
print("rho3:", rho3)

print("tao:", tao)

print("alpha:", alpha)

for i in range(ITER):
    print("ITER:", i)
    loss, acc = inference(X_train, y_train, s_new, u, w, b, v, c)
    train_acc[i] = acc
    print("training acc:", acc)
    train_loss[i] = loss
    print("training loss:", loss)
    curtime = time.time()
    o = common.update_o(y_train, lambda3, o, v, s_new, c, rho3, alpha)
    c = common.update_c(lambda3, o, v, s_new, c, rho3, alpha)
    v = common.update_v(lambda3, o, v, s_new, c, rho3, alpha)
    s_new = common.update_s_new(lambda2, a, rho2, lambda3, o, v, s_new, c, rho3, tao, alpha)
    a = common.update_a(lambda1, a, u, X_train, w, s_old, b, rho1, lambda2, rho2, s_new, alpha)
    b = common.update_b(lambda1, a, u, X_train, w, s_old, b, rho1, tao, alpha)
    s_old = common.update_s_last(lambda1, rho1, a, u, X_train, w, s_old, b, tao, alpha)
    w = common.update_w(lambda1, a, u, X_train, w, s_old, b, rho1, tao, alpha)
    u = common.update_u(lambda1, a, u, X_train, w, s_old, b, rho1, tao, alpha)

    u = common.update_u(lambda1, a, u, X_train, w, s_old, b, rho1, tao, alpha)
    w = common.update_w(lambda1, a, u, X_train, w, s_old, b, rho1, tao, alpha)
    s_old = common.update_s_last(lambda1, rho1, a, u, X_train, w, s_old, b, tao, alpha)
    b = common.update_b(lambda1, a, u, X_train, w, s_old, b, rho1, tao, alpha)
    a = common.update_a(lambda1, a, u, X_train, w, s_old, b, rho1, lambda2, rho2, s_new, alpha)
    s_new = common.update_s_new(lambda2, a, rho2, lambda3, o, v, s_new, c, rho3, tao, alpha)
    v = common.update_v(lambda3, o, v, s_new, c, rho3, alpha)
    c = common.update_c(lambda3, o, v, s_new, c, rho3, alpha)
    o = common.update_o(y_train, lambda3, o, v, s_new, c, rho3, alpha)

    lambda1 = common.update_lambda1(lambda1, rho1, a, u, X_train, w, s_old, b)
    lambda2 = common.update_lambda2(lambda2, rho2, s_new, a)
    lambda3 = common.update_lambda3(lambda3, rho3, o, v, s_new, c)
    print("Time per iteration:", time.time() - curtime)

    # test_loss, test_acc = inference(X_test, y_test, s_new, u, w, b, v, c)
    # test_acc[i] = test_acc
    # loss_test[i] = test_loss
    # print("test acc:", test_acc)
    # print("test loss:", test_loss)

    # prediction = common.softmax(o)
    # # correct_prediction = np.equal(np.argmax(y_train, 1), np.argmax(prediction, 1))
    # acc = np.sum(np.equal(np.argmax(prediction, axis=0), np.argmax(y_train, axis=0))) / nums
    # train_acc[i] = acc
    # print("training acc:", acc)
    #
    # loss = common.cross_entropy_with_softmax(y_train, o)
    # loss = np.mean(loss)
    # print("training loss:", loss)

