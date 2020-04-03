import numpy as np
import common

# data I/O

data = open('text.txt', 'r').read() # should be simple plain text file

# use set() to count the vacab size
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))

# dictionary to convert char to idx, idx to char
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 128 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
## RNN/LSTM
## this is not LSTM, is the simple basic RNN
## # update the hidden state
## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
## # compute the output vector
## y = np.dot(self.W_hy, self.h)
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden (u)
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden (w)
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output  (v)
bh = np.zeros((hidden_size, 1)) # hidden bias (b)
by = np.zeros((vocab_size, 1)) # output bias  (c)


## compute loss, derivative
## cross-entropy loss is used
## actually, here the author use cross-entropy as error,
## but in the backpropagation the author use sum of squared error (Quadratic cost) to do back propagation.
## be careful about this trick.
## this is because the output layer is a linear layer.
## TRICK: Using the quadratic cost when we have linear neurons in the output layer, z[i] = a[i]
def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    a = {}
    ## record each hidden state of
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass for each training data point
    # for t in xrange(len(inputs)):
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        a[t] = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh
        ## hidden state, using previous hidden state hs[t-1]
        hs[t] = np.tanh(a[t])
        ## unnormalized log probabilities for next chars
        ys[t] = np.dot(Why, hs[t]) + by
        ## probabilities for next chars, softmax
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        ## softmax (cross-entropy loss)
        loss += -np.log(ps[t][targets[t], 0])

    loss = loss / seq_length

    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    da = {}
    do = {}
    ds = {}
    for t in reversed(range(len(inputs))):
        ## compute derivative of error w.r.t the output probabilites
        ## dE/dy[j] = y[j] - t[j] namely, dy
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # backprop into y
        do[t] = dy

        ## output layer doesnot use activation function, so no need to compute the derivative of error with regard to the net input
        ## of output layer.
        ## then, we could directly compute the derivative of error with regard to the weight between hidden layer and output layer.
        ## dE/dy[j]*dy[j]/dWhy[j,k] = dE/dy[j] * h[k]
        dWhy += np.dot(dy, hs[t].T)
        dby += dy

        ## backprop into h
        ## derivative of error with regard to the output of hidden layer
        ## derivative of H, come from output layer y and also come from H(t+1), the next time H
        dh = np.dot(Why.T, dy) + dhnext
        ds[t] = dh
        ## backprop through tanh nonlinearity
        ## derivative of error with regard to the input of hidden layer
        ## dtanh(x)/dx = 1 - tanh(x) * tanh(x)
        ## dhraw is the bp derivative of da[t]
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        da[t] = np.copy(dhraw)

        ## derivative of the error with regard to the weight between input layer and hidden layer
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        ## derivative of the error with regard to H(t+1)
        ## or derivative of the error of H(t-1) with regard to H(t)
        dhnext = np.dot(Whh.T, dhraw)

    # for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    #     np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1], a, hs, ys, da, do, ds

## given a hidden RNN state, and a input char id, predict the coming n chars
def sample(h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """

    ## a one-hot vector
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1

    ixes = []
    for t in range(n):
        ## self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        ## y = np.dot(self.W_hy, self.h)
        y = np.dot(Why, h) + by
        ## softmax
        p = np.exp(y) / np.sum(np.exp(y))
        ## sample according to probability distribution
        ix = np.random.choice(range(vocab_size), p=p.ravel())

        ## update input x
        ## use the new sampled result as last input, then predict next char again.
        x = np.zeros((vocab_size, 1))
        x[ix] = 1

        ixes.append(ix)

    return ixes

def inference(Wxh, Whh, Why, bh, by, inputs, targets, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    a = {}
    ## record each hidden state of
    hs[-1] = np.copy(hprev)
    loss = 0
    # forward pass for each training data point
    # for t in xrange(len(inputs)):
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        a[t] = np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh
        ## hidden state, using previous hidden state hs[t-1]
        hs[t] = np.tanh(a[t])
        ## unnormalized log probabilities for next chars
        ys[t] = np.dot(Why, hs[t]) + by
        ## probabilities for next chars, softmax
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        ## softmax (cross-entropy loss)
        loss += -np.log(ps[t][targets[t], 0])

    # loss = loss / len(inputs)

    return loss


## iterator counter
n = 0
## data pointer
p = 0

mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size) # *seq_length # loss at iteration 0

rho1 = 0.025
rho2 = 0.025
rho3 = 0.025
lambda1 = np.zeros((hidden_size, 1))
lambda2 = np.zeros((hidden_size, 1))
lambda3 = np.zeros((vocab_size, 1))
loss_ = []

mu = 0.00000001
alpha = 10 #seq_length

## main loop
while True:
    # prepare inputs (we're sweeping from left to right in steps seq_length long)
    if p + seq_length + 1 >= len(data) or n == 0:     # p+seq_length+1>=len(data) exclude the dataset
        # reset RNN memory
        ## hprev is the hiddden state of RNN
        hprev = np.zeros((hidden_size, 1))
        # go from start of data
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p : p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1 : p + seq_length + 1]]

    # sample from the model now and then
    # if n % 100 == 0:
    #     sample_ix = sample(hprev, inputs[0], 200)
    #     txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    #     print ('---- sample -----')
    #     print ('----\n %s \n----' % (txt, ))

    # forward seq_length characters through the net and fetch gradient
    # a: middle state after the activation function
    # hs: state(s)
    # ys: output(o)
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev, a, hs, ys, da, do, ds = lossFun(inputs, targets, hprev)
    loss_.append(loss)
    ## author using Adagrad(a kind of gradient descent)
    smooth_loss = smooth_loss * 0.99 + loss * 0.01
    # smooth_loss = loss
    # if n % 100 == 0:
    #   print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress

    print('iter %d, loss: %f' % (n, smooth_loss))

    N = len(inputs)
    # T = 100
    # # for i in range(T):
    ys = common.update_o(ys, do, inputs, lambda3, Why, hs, by, rho3, mu, alpha)
    by = common.update_c(by, dby, inputs, lambda3, ys, Why, hs, rho3, mu, alpha)
    Why = common.update_v(Why, dWhy, inputs, lambda3, ys, hs, by, rho3, mu, alpha)
    hs = common.update_s(hs, ds, inputs, lambda1, a, Wxh, Whh, bh, rho1, lambda2, rho2, lambda3, ys, Why, by, rho3, mu,
                         alpha)
    a = common.update_a(a, da, inputs, lambda1, Wxh, Whh, hs, bh, rho1, lambda2, rho2, mu)
    bh = common.update_b(bh, dbh, inputs, lambda1, a, Wxh, Whh, hs, rho1, mu, alpha)
    Whh = common.update_w(Whh, dWhh, inputs, lambda1, a, Wxh, hs, bh, rho1, mu, alpha)
    Wxh = common.update_u(Wxh, dWxh, inputs, lambda1, a, Whh, hs, bh, rho1, mu, alpha)

    Wxh = common.update_u(Wxh, dWxh, inputs, lambda1, a, Whh, hs, bh, rho1, mu, alpha)
    Whh = common.update_w(Whh, dWhh, inputs, lambda1, a, Wxh, hs, bh, rho1, mu, alpha)
    bh = common.update_b(bh, dbh, inputs, lambda1, a, Wxh, Whh, hs, rho1, mu, alpha)
    a = common.update_a(a, da, inputs, lambda1, Wxh, Whh, hs, bh, rho1, lambda2, rho2, mu)
    hs = common.update_s(hs, ds, inputs, lambda1, a, Wxh, Whh, bh, rho1, lambda2, rho2, lambda3, ys, Why, by, rho3, mu, alpha)
    Why = common.update_v(Why, dWhy, inputs, lambda3, ys, hs, by, rho3, mu, alpha)
    by = common.update_c(by, dby, inputs, lambda3, ys, Why, hs, rho3, mu, alpha)
    ys = common.update_o(ys, do, inputs, lambda3, Why, hs, by, rho3, mu, alpha)

    lambda1 = common.update_lambda1(inputs, lambda1, rho1, a, Wxh, Whh, hs, bh)
    lambda2 = common.update_lambda2(inputs, lambda2, rho2, hs, a)
    lambda3 = common.update_lambda3(inputs, lambda3, rho3, ys, Why, hs, by)

    # loss = inference(Wxh, Whh, Why, bh, by, inputs, targets, hprev)
    # print("iter %d, loss: %f:" % (n, loss))

    # TODO: whether to choose a changing rule for rho1, rho2, rho3, mu and alpha
    # if n>2 and loss_[n] <= loss_[n-1] and loss_[n-1] <= loss_[n-2] and loss_[n-2] <= loss_[n-3]:
    # # # alpha = np.minimum(100, alpha * 10)
    #     rho1 = np.maximum(0.025, rho1/10)
    #     rho2 = np.maximum(0.025, rho2/10)
    #     rho3 = np.maximum(0.025, rho3/10)
    #     mu = mu / 10





    # perform parameter update with Adagrad
    ## parameter update for Adagrad is different from gradient descent parameter update
    ## need to learn what is Adagrad exactly is.
    ## seems using weight matrix, derivative of weight matrix and a memory matrix, update memory matrix each iteration
    ## memory is the accumulation of each squared derivatives in each iteration.
    ## mem += dparam * dparam
    # for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
    #                               [dWxh, dWhh, dWhy, dbh, dby],
    #                               [mWxh, mWhh, mWhy, mbh, mby]):
    #   mem += dparam * dparam
    #   ## learning_rate is adjusted by mem, if mem is getting bigger, then learning_rate will be small
    #   ## gradient descent of Adagrad
    #   param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

    p += seq_length # move data pointer
    # p = 0
    n += 1 # iteration counter
    #
    if n>=1000:
        break