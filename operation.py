## all of the update rule in ADMM RNN
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import numpy as np
# import tensorflow as tf
import torch


# tanh
def tanh(x):
    return torch.tanh(x)


# return the derivative of tanh()
def der_tanh(x):
    # temp = tanh(tran) * tanh(x)
    # temp = np.mat(tanh(x.T)) * np.mat(tanh(x))
    temp = x ** 2
    res = 1 - temp
    return res


def der_ce_softmax(label, output):
    prediction = softmax(output)
    res = prediction - label
    return res

# softmax
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), axis=0)


# cross-entropy loss function
def cross_entropy(label, prob):
    loss = -torch.sum(label * torch.log(prob))
    return loss


# crosss_entropy with softmax
def cross_entropy_with_softmax(label, output):
    prediction = softmax(output)
    loss = cross_entropy(label, prediction)
    return loss


# return relu
def relu(x):
    return torch.maximum(x, 0)


# return the value of phi
def phi(a, u, x, w, s_old, b, s_new, lambda1, lambda2, lambda3, o, v, rho1, rho2, rho3):
    temp1 = a - torch.matmul(u, x) - torch.matmul(w, s_old) - b + lambda1 / rho1
    temp1 = rho1 / 2 * torch.sum(temp1 * temp1)
    temp2 = s_new - tanh(a) + lambda2 / rho2
    temp2 = rho2 / 2 * torch.sum(temp2 * temp2)
    temp3 = o - torch.matmul(v, s_new) + lambda3 / rho3
    temp3 = rho3 / 2 * torch.sum(temp3 * temp3)
    res = temp1 + temp2 + temp3
    return res


# return lagrange function
def lagrange(loss, mu, a, u, x, w, s_old, b):
    return


##### the derivative of phi with regard to all the parameters
# return the derivative of phi with regard to u
def phi_u(lambda1, rho1, a, u, x, w, s_old, b, tao = 1):
    temp1 = a - torch.matmul(u, x) - torch.matmul(w, s_old) - b - lambda1 / rho1
    temp = rho1 * torch.matmul(temp1, x.T)
    res = u - temp / tao
    return res


# return the derivative of phi with regard to w
def phi_w(lambda1, rho1, a, u, x, w, s_old, b, tao = 1):
    temp1 = a - torch.matmul(u, x) - torch.matmul(w, s_old) - b - lambda1 / rho1
    temp = rho1 * torch.matmul(temp1, s_old.T)
    res = w - temp / tao
    return res


# return the derivative of phi with regard to s_old
def phi_s_last(lambda1, rho1, a, u, x, w, s_last_old, b, tao):
    temp1 = a - torch.matmul(u, x) - torch.matmul(w, s_last_old) - b - lambda1 / rho1
    temp = rho1 * torch.matmul(w.T, temp1)
    res = s_last_old - temp / tao
    return res


# return the derivative of phi with regard to b
def phi_b(lambda1, rho1, a, u, x, w, s_old, b, tao):
    temp1 = a - torch.matmul(u, x) - torch.matmul(w, s_old) - lambda1 / rho1
    res = temp1
    return res


# return the derivative of phi with regard to a
def phi_a(lambda1, rho1, a, u, x, w, s_old, b, lambda2, rho2, s_new):
    tao = 1
    temp1 = a - (torch.matmul(u, x) + torch.matmul(w, s_old) + b + lambda1 / rho1)
    temp1 = rho1 * temp1
    der = -der_tanh(a)
    temp2 = (s_new - tanh(a) - lambda2 / rho2) * der
    temp2 = rho2 * temp2
    res = a - (temp1 / tao + temp2 / tao)
    return res


# return the derivative of phi with regard to s_new
def phi_s_new(lambda2, rho2, s_new, a, lambda3, rho3, o, v, c, tao):
    temp1 = rho2 * (s_new - (tanh(a) + lambda2 / rho2))
    temp2 = o - torch.matmul(v, s_new) - c - lambda3 / rho3
    vt = v.T
    temp = temp1 + rho3 * torch.matmul(vt, temp2)
    res = s_new - temp / tao
    return res


# return the derivative of phi with regard to v
def phi_v(lambda3, rho3, o, v, s_new, c, tao):
    temp1 = o - torch.matmul(v, s_new) - c - lambda3 / rho3
    st = s_new.T
    temp = rho3 * torch.matmul(temp1, st)
    res = v - temp / tao
    return res


# return the derivative of phi with regard to c
def phi_c(lambda3, rho3, o, v, s_new, c):
    temp1 = o - torch.matmul(v, s_new) - lambda3 / rho3
    res = temp1
    return res


# return the derivative of phi with regard to o
def phi_o(lambda3, rho3, o, v, s_new, c):
    temp1 = torch.matmul(v, s_new) + c + lambda3 / rho3
    res = temp1
    return res


### update rules
# return updated u corresponding to x(input)
def update_u(lambda1, a, u_old, x, w, s_old, b, rho1, tao, alpha=1):
    u_gradient = phi_u(lambda1, rho1, a, u_old, x, w, s_old, b, tao)
    u_new = u_gradient / alpha
    return u_new


# return updated w corresponding to s(state)
def update_w(lambda1, a, u, x, w_old, s_old, b, rho1, tao, alpha=1):
    w_gradient = phi_w(lambda1, rho1, a, u, x, w_old, s_old, b, tao)
    w_new = w_gradient / alpha
    return w_new


# return updated bias
def update_b(lambda1, a, u, x, w, s_old, b_old, rho1, tao, alpha=1):
    b_gradient = phi_b(lambda1, rho1, a, u, x, w, s_old, b_old, tao)
    b_new = b_gradient / alpha
    return b_new


# return updated a
def update_a(lambda1, a_old, u, x, w, s_old, b, rho1, lambda2, rho2, s_new, alpha=1):
    a_gradient = phi_a(lambda1, rho1, a_old, u, x, w, s_old, b, lambda2, rho2, s_new)
    a_new = a_gradient / alpha
    return a_new

# return updated state
def update_s_last(lambda1, rho1, a, u, x, w, s_last_old, b, tao, alpha=1):
    s_gradient = phi_s_last(lambda1, rho1, a, u, x, w, s_last_old, b, tao)
    s_update = s_gradient / alpha
    return s_update

# return updated state
def update_s_new(lambda2, a, rho2, lambda3, o, v, s_new, c, rho3, tao, alpha=1):
    s_gradient = phi_s_new(lambda2, rho2, s_new, a, lambda3, rho3, o, v, c, tao)
    s_update = s_gradient / alpha
    return s_update


# return updated v corresponding to output
def update_v(lambda3, o, v_old, s_new, c, rho3, tao, alpha=1):
    v_gradient = phi_v(lambda3, rho3, o, v_old, s_new, c, tao)
    v_new = v_gradient / alpha
    return v_new


# return updated c corresponding to output
def update_c(lambda3, o, v, s_new, c_old, rho3, alpha=1):
    c_gradient = phi_c(lambda3, rho3, o, v, s_new, c_old)
    c_new = c_gradient / alpha
    return c_new


# return update o(output)
def update_o(label, lambda3, o_old, v, s_new, c, rho3, alpha=1):
    o_gradient = phi_o(lambda3, rho3, o_old, v, s_new, c)
    der = der_ce_softmax(label, o_old)
    temp = o_gradient - der / rho3
    o_new = temp / alpha
    return o_new


# update of lambda1
def update_lambda1(lambda1, rho1, a, u, x, w, s_old, b):
    temp = a - torch.matmul(u, x) - torch.matmul(w, s_old) - b
    lambda_new = lambda1 - rho1 * temp
    return lambda_new


# update of lambda2
def update_lambda2(lambda2, rho2, s_new, a):
    temp = s_new - tanh(a)
    lambda_new = lambda2 - rho2 * temp
    return lambda_new


# update of lambda3
def update_lambda3(lambda3, rho3, o, v, s_new, c):
    temp = o - torch.matmul(v, s_new) - c
    lambda_new = lambda3 - rho3 * temp
    return lambda_new
