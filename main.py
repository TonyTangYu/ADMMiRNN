import torch
from torch import Tensor
import common as common
import math
# hyperparameters
hidden_size = 128  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for
learning_rate = 1e-1


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    a = {}
    ## record each hidden state of
    hs[-1] = hprev.clone()
    loss = 0
    # forward pass for each training data point
    # for t in xrange(len(inputs)):
    for t in range(len(inputs)):
        xs[t] = torch.zeros((vocab_size, 1))  # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        a[t] = torch.mm(Wxh, xs[t]) + torch.mm(Whh, hs[t - 1]) + bh
        ## hidden state, using previous hidden state hs[t-1]
        hs[t] = torch.tanh(a[t])
        ## unnormalized log probabilities for next chars
        ys[t] = torch.mm(Why, hs[t]) + by
        ## probabilities for next chars, softmax
        ps[t] = torch.exp(ys[t]) / torch.sum(torch.exp(ys[t]))
        ## softmax (cross-entropy loss)
        loss += -torch.log(ps[t][targets[t], 0])

    loss = loss / seq_length

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
    dbh, dby = torch.zeros_like(bh), torch.zeros_like(by)
    dhnext = torch.zeros_like(hs[0])
    da = {}
    do = {}
    ds = {}
    for t in reversed(range(len(inputs))):
        ## compute derivative of error w.r.t the output probabilites
        ## dE/dy[j] = y[j] - t[j] namely, dy
        dy = ps[t].clone()
        dy[targets[t]] -= 1  # backprop into y
        do[t] = dy

        ## output layer doesnot use activation function, so no need to compute the derivative of error with regard to the net input
        ## of output layer.
        ## then, we could directly compute the derivative of error with regard to the weight between hidden layer and output layer.
        ## dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
        dWhy += torch.mm(dy, hs[t].T)
        dby += dy

        ## backprop into h
        ## derivative of error with regard to the output of hidden layer
        ## derivative of H, come from output layer y and also come from H(t+1), the next time H
        dh = torch.mm(Why.T, dy) + dhnext
        ds[t] = dh
        ## backprop through tanh nonlinearity
        ## derivative of error with regard to the input of hidden layer
        ## dtanh(x)/dx = 1 - tanh(x) * tanh(x)
        ## dhraw is the bp derivative of da[t]
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        da[t] = dhraw.clone()

        ## derivative of the error with regard to the weight between input layer and hidden layer
        dWxh += torch.mm(dhraw, xs[t].T)
        dWhh += torch.mm(dhraw, hs[t - 1].T)
        ## derivative of the error with regard to H(t+1)
        ## or derivative of the error of H(t-1) with regard to H(t)
        dhnext = torch.mm(Whh.T, dhraw)

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1], a, hs, ys, da, do, ds


if __name__ == '__main__':
    #########################################################
    data = open('text.txt', 'r').read()
    chars = list(set(data))  # 这里还是挺巧妙的 可以直接获得个数和对应的序号
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))

    # dictionary to convert char to idx, idx to char
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    #########################################################

    Wxh = torch.rand(hidden_size, vocab_size) * 0.01  # input to hidden (u)
    Whh = torch.rand(hidden_size, hidden_size) * 0.01  # hidden to hidden (w)
    Why = torch.rand(vocab_size, hidden_size) * 0.01  # hidden to output  (v)
    bh = torch.zeros((hidden_size, 1))  # hidden bias (b)
    by = torch.zeros((vocab_size, 1))  # output bias  (c)


    #########################################################
    ## iterator counter
    n = 0
    ## data pointer
    p = 0

    mWxh, mWhh, mWhy = torch.zeros_like(Wxh), torch.zeros_like(Whh), torch.zeros_like(Why)
    mbh, mby = torch.zeros_like(bh), torch.zeros_like(by)  # memory variables for Adagrad
    smooth_loss = -math.log(1.0 / vocab_size)  # *seq_length # loss at iteration 0

    rho1 = 0.025
    rho2 = 0.025
    rho3 = 0.025
    lambda1 = torch.zeros((hidden_size, 1))
    lambda2 = torch.zeros((hidden_size, 1))
    lambda3 = torch.zeros((vocab_size, 1))
    loss_ = []

    mu = 0.00000001
    alpha = 10  # seq_length

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= len(data) or n == 0:  # p+seq_length+1>=len(data) exclude the dataset
            # reset RNN memory
            ## hprev is the hiddden state of RNN
            hprev = torch.zeros((hidden_size, 1))
            # go from start of data
            p = 0

        inputs = [char_to_ix[ch] for ch in data[p: p + seq_length]]
        targets = [char_to_ix[ch] for ch in data[p + 1: p + seq_length + 1]]

        loss, dWxh, dWhh, dWhy, dbh, dby, hprev, a, hs, ys, da, do, ds = lossFun(inputs, targets, hprev)
        loss_.append(loss)
        ## author using Adagrad(a kind of gradient descent)
        smooth_loss = smooth_loss * 0.99 + loss * 0.01

        print('iter %d, loss: %f' % (n, smooth_loss))

        N = len(inputs)

        ys = common.update_o(ys, do, inputs, lambda3, Why, hs, by, rho3, mu, alpha)
        by = common.update_c(by, dby, inputs, lambda3, ys, Why, hs, rho3, mu, alpha)
        Why = common.update_v(Why, dWhy, inputs, lambda3, ys, hs, by, rho3, mu, alpha)
        hs = common.update_s(hs, ds, inputs, lambda1, a, Wxh, Whh, bh, rho1, lambda2, rho2, lambda3, ys, Why, by, rho3,
                             mu,
                             alpha)
        a = common.update_a(a, da, inputs, lambda1, Wxh, Whh, hs, bh, rho1, lambda2, rho2, mu)
        bh = common.update_b(bh, dbh, inputs, lambda1, a, Wxh, Whh, hs, rho1, mu, alpha)
        Whh = common.update_w(Whh, dWhh, inputs, lambda1, a, Wxh, hs, bh, rho1, mu, alpha)
        Wxh = common.update_u(Wxh, dWxh, inputs, lambda1, a, Whh, hs, bh, rho1, mu, alpha)

        Wxh = common.update_u(Wxh, dWxh, inputs, lambda1, a, Whh, hs, bh, rho1, mu, alpha)
        Whh = common.update_w(Whh, dWhh, inputs, lambda1, a, Wxh, hs, bh, rho1, mu, alpha)
        bh = common.update_b(bh, dbh, inputs, lambda1, a, Wxh, Whh, hs, rho1, mu, alpha)
        a = common.update_a(a, da, inputs, lambda1, Wxh, Whh, hs, bh, rho1, lambda2, rho2, mu)
        hs = common.update_s(hs, ds, inputs, lambda1, a, Wxh, Whh, bh, rho1, lambda2, rho2, lambda3, ys, Why, by, rho3,
                             mu,
                             alpha)
        Why = common.update_v(Why, dWhy, inputs, lambda3, ys, hs, by, rho3, mu, alpha)
        by = common.update_c(by, dby, inputs, lambda3, ys, Why, hs, rho3, mu, alpha)
        ys = common.update_o(ys, do, inputs, lambda3, Why, hs, by, rho3, mu, alpha)

        lambda1 = common.update_lambda1(inputs, lambda1, rho1, a, Wxh, Whh, hs, bh)
        lambda2 = common.update_lambda2(inputs, lambda2, rho2, hs, a)
        lambda3 = common.update_lambda3(inputs, lambda3, rho3, ys, Why, hs, by)

        # TODO: whether to choose a changing rule for rho1, rho2, rho3, mu and alpha

        p += seq_length  # move data pointer
        # p = 0
        n += 1  # iteration counter
        #
        if n >= 10000:
            break



