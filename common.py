## all of the update rule in ADMM RNN
import torch


# tanh
def tanh(x):
    return torch.tanh(x)


# return the derivative of tanh()
def der_tanh(x):
    temp = x ** 2
    res = 1 - temp
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
    return torch.max(x, 0)


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
def phi_u(lambda1, rho1, a, u, inputs, w, s, b, mu):
    vocab_size = u.shape[1]
    x = {}
    tao = 1
    temp1 = torch.zeros_like(u)
    # temp2 = torch.zeros_like(u)
    # mu = 0.001
    for t in range(len(inputs) - 1):
        x[t] = torch.zeros((vocab_size, 1))
        x[t][inputs[t]] = 1
        temp= a[t] - torch.mm(u, x[t]) - torch.mm(w, s[t-1]) - b
        temp1 = temp1 + torch.mm(temp, x[t].T)
        # temp2 = temp2 + torch.mm(temp, x[t].T)
    temp = mu * temp1
    N = len(inputs) - 1
    x[N] = torch.zeros((vocab_size, 1))
    x[N][inputs[N]] = 1
    temp2 = a[N] - torch.mm(u, x[N]) - torch.mm(w, s[N-1]) - b - lambda1 / rho1
    final = rho1 * torch.mm(temp2, x[N].T)
    res = u - (temp + final) / tao
    return res


# return the derivative of phi with regard to w
def phi_w(lambda1, rho1, a, u, inputs, w_old, s, b, mu):
    vocab_size = u.shape[1]
    x = {}
    tao = 1
    temp1 = torch.zeros_like(w_old)
    # temp2 = torch.zeros_like(w_old)
    mu = 0.1
    for t in range(len(inputs) - 1):
        x[t] = torch.zeros((vocab_size, 1))
        x[t][inputs[t]] = 1
        temp = a[t] - torch.mm(u, x[t]) - torch.mm(w_old, s[t-1]) - b
        temp1 = temp1 + torch.mm(temp, s[t-1].T)
        # temp2 = temp2 + torch.mm(temp, s[t-1].T)
    temp2 = mu * temp1
    N = len(inputs) - 1
    x[N] = torch.zeros((vocab_size, 1))
    x[N][inputs[N]] = 1
    temp = a[N] - torch.mm(u, x[N]) - torch.mm(w_old, s[N - 1]) - b - lambda1 / rho1
    final = rho1 * torch.mm(temp, s[N-1].T)
    res = w_old - (temp2 + final) / tao
    return res


# return the derivative of phi with regard to b
def phi_b(lambda1, rho1, a, u, inputs, w, s, b_old, mu):
    vocab_size = u.shape[1]
    x = {}
    temp1 = torch.zeros_like(b_old)
    # mu = 0.1
    for t in range(len(inputs) - 1):
        x[t] = torch.zeros((vocab_size, 1))
        x[t][inputs[t]] = 1
        temp = a[t] - torch.mm(u, x[t]) - torch.mm(w, s[t-1])
        temp1 = temp1 + temp
    temp = mu * temp1
    N = len(inputs) - 1
    x[N] = torch.zeros((vocab_size, 1))
    x[N][inputs[N]] = 1
    temp_final = a[N] - torch.mm(u, x[N]) - torch.mm(w, s[N-1]) - lambda1 / rho1
    final = rho1 * temp_final
    res = (temp + final) / (rho1 + mu*N)
    return res


# return the derivative of phi with regard to a
def phi_a(lambda1, a_old, u, inputs, w, s, b, rho1, lambda2, rho2, mu):
    vocab_size = u.shape[1]
    N = len(inputs) - 1
    x = {}
    res = {}
    tao = 1
    for t in range(N):
        x[t] = torch.zeros((vocab_size, 1))
        x[t][inputs[t]] = 1
        # temp1 = a_old[t] - torch.mm(u, x[t]) - torch.mm(w, s[t-1]) - b
        temp1 = a_old[t] - (torch.mm(u, x[t]) + torch.mm(w, s[t - 1]) + b)
        der = -der_tanh(a_old[t])
        temp2 = (s[t]-tanh(a_old[t])) * der      # torch.mm(s[t]-tanh(a_old[t]), der)
        res[t]= a_old[t] + (mu / tao) * (temp1 + temp2)
    x[N] = torch.zeros((vocab_size, 1))
    x[N][inputs[N]] = 1
    # temp3 = a_old[N] - torch.mm(u, x[N]) - torch.mm(w, s[N-1]) - b - lambda1 / rho1
    temp1 = a_old[N] - (torch.mm(u, x[N]) + torch.mm(w, s[N - 1]) + b + lambda1 / rho1)
    temp3 = rho1 * temp1
    temp2 = s[N] - tanh(a_old[N]) - lambda2 / rho2
    der = -der_tanh(a_old[N])
    temp4 = rho2 * temp2 * der
    # res[N] = lambda1 + rho1 * temp3 + lambda2 - rho2 * temp4 * der
    res[N] = a_old[N] + (temp3 + temp4) / tao
    return res


# return the derivative of phi with regard to s_old
def phi_s(lambda1, s_old, a, u, inputs, w, b, rho1, lambda2, rho2, lambda3, o, v, c, rho3, mu):
    vocab_size = u.shape[1]
    N = len(inputs) - 1
    x = {}
    res = {}
    tao = 1
    res[-1] = s_old[-1].clone()
    for t in range(N):
        x[t] = torch.zeros((vocab_size, 1))
        x[t][inputs[t]] = 1
        # temp1 = a[t] - torch.mm(u, x[t]) - torch.mm(w, s_old[t - 1]) - b
        # temp2 = s_old[t] - tanh(a[t]) - lambda2 / rho2
        temp1 = mu * tanh(a[t])
        temp2 = (N-1) / N * s_old[t]
        temp3 = o[t] - torch.mm(v, s_old[t]) - c
        temp= temp1 + temp2 - mu * torch.mm(v.T, temp3)
        div = mu + tao
        res[t] = temp / div
    x[N] = torch.zeros((vocab_size, 1))
    x[N][inputs[N]] = 1
    temp2_final = tanh(a[N]) + lambda2 / rho2
    temp3_final = o[N] - torch.mm(v, s_old[N]) - c - lambda3 / rho3
    temp_final = rho2 * temp2_final + tao * s_old[N] - rho3 * torch.mm(v.T, temp3_final)
    res[N] = temp_final / (rho2 + tao)
    return res


# return the derivative of phi with regard to v
def phi_v(lambda3, rho3, inputs, o, v, s, c, mu):
    temp1 = torch.zeros_like(v)
    # temp2 = torch.zeros_like(v)
    N = len(inputs) - 1
    tao = 1
    for t in range(N):
        temp = o[t] - torch.mm(v, s[t]) - c
        temp1 = temp1 + torch.mm(temp, s[t].T)
        # temp2 = temp2 + torch.mm(temp, s[t].T)
    temp = mu * temp1
    temp2 = o[N] - torch.mm(v, s[N]) - c - lambda3 / rho3
    final = rho3 * torch.mm(temp2, s[N].T)
    res = v - (temp + final) / tao
    return res


# return the derivative of phi with regard to c
def phi_c(lambda3, rho3, inputs, o, v, s, c, mu):
    temp1 = torch.zeros_like(c)
    N = len(inputs) - 1
    for t in range(N):
        temp = o[t] - torch.mm(v, s[t])
        temp1 = temp1 + temp
    temp = mu * temp1
    temp2 = o[N] - torch.mm(v, s[N]) - lambda3 / rho3
    final = rho3 * temp2
    res = (temp + final) / (rho3 + mu*(N-1))
    return res


# return the derivative of phi with regard to o
def phi_o(do, inputs, lambda3, rho3, o, v, s, c, mu):
    N = len(inputs) - 1
    res = {}
    for t in range(N):
        # temp = o[t] - torch.mm(v, s[t]) - c - lambda3 / rho3
        temp = torch.mm(v, s[t]) + c
        res[t] = temp - do[t] / mu
    temp = torch.mm(v, s[N]) + c + lambda3 / rho3
    res[N] = temp - do[N] / rho3
    return res


##
### update rules
# return updated u corresponding to x(input)
def update_u(u_old, du, inputs, lambda1, a, w, s, b, rho1, mu, alpha=1):
    u_gradient = phi_u(lambda1, rho1, a, u_old, inputs, w, s, b, mu)
    du = 0
    # u_new = u_old - du - u_gradient / alpha
    u_new = u_gradient / alpha
    return u_new

# return updated w corresponding to s(state)
def update_w(w_old, dw, inputs, lambda1, a, u, s, b, rho1, mu, alpha=1):
    w_gradient = phi_w(lambda1, rho1, a, u, inputs, w_old, s, b, mu)
    dw = 0
    # w_new = w_old - dw - w_gradient / alpha
    w_new = w_gradient / alpha
    return w_new


# return updated bias
def update_b(b_old, db, inputs, lambda1, a, u, w, s, rho1, mu, alpha=1):
    b_gradient = phi_b(lambda1, rho1, a, u, inputs, w, s, b_old, mu)
    db = 0
    # b_new = b_old - db - b_gradient / alpha
    b_new = b_gradient / alpha
    return b_new


# return updated a
def update_a(a_old, da, inputs, lambda1, u, w, s, b, rho1, lambda2, rho2, mu, alpha=1):
    a_gradient = phi_a(lambda1, a_old, u, inputs, w, s, b, rho1, lambda2, rho2, mu)
    a_new = {}
    N = len(inputs)
    for t in range(N):
        da[t] = 0
        # a_new[t] = a_old[t] - da[t] - a_gradient[t] / alpha
        a_new[t] = a_gradient[t] / alpha
    return a_new


# return updated state
def update_s(s_old, ds, inputs, lambda1, a, u, w, b, rho1, lambda2, rho2, lambda3, o, v, c, rho3, mu, alpha=1):
    s_gradient = phi_s(lambda1, s_old, a, u, inputs, w, b, rho1, lambda2, rho2, lambda3, o, v, c, rho3, mu)
    s_new = {}
    s_new[-1] = s_gradient[-1]
    N = len(inputs)
    for t in range(N):
        ds[t] = 0
        # s_new[t] = s_old[t] - ds[t] - s_gradient[t] / alpha
        s_new[t] = s_gradient[t] / alpha
    return s_new

# return updated v corresponding to output
def update_v(v_old, dv, inputs, lambda3, o, s, c, rho3, mu, alpha=1):
    v_gradient = phi_v(lambda3, rho3, inputs, o, v_old, s, c, mu)
    dv = 0
    # v_new = v_old - dv - v_gradient / alpha
    v_new = v_gradient / alpha
    return v_new


# return updated c corresponding to output
def update_c(c_old, dc, inputs, lambda3, o, v, s, rho3, mu, alpha=1):
    c_gradient = phi_c(lambda3, rho3, inputs, o, v, s, c_old, mu)
    dc = 0
    # c_new = c_old - dc - c_gradient / alpha
    c_new = c_gradient / alpha
    return c_new


# return update o(output)
def update_o(o_old, do, inputs, lambda3, v, s, c, rho3, mu, alpha=1):
    o_gradient = phi_o(do, inputs, lambda3, rho3, o_old, v, s, c, mu)
    o_new = {}
    N = len(inputs)
    for t in range(N):
        # o_new[t] = o_old[t] - do[t] - o_gradient[t] / alpha
        o_new[t] = o_gradient[t] / alpha
    return o_new


# update of lambda1
def update_lambda1(inputs, lambda1, rho1, a, u, w, s, b):
    N = len(inputs) - 1
    x = {}
    vocab_size = u.shape[1]
    x[N] = torch.zeros((vocab_size, 1))
    x[N][inputs[N]] = 1
    temp = a[N] - torch.mm(u, x[N]) - torch.mm(w, s[N-1]) - b
    lambda_new = (lambda1 - rho1 * temp) / 10
    return lambda_new


# update of lambda2
def update_lambda2(inputs, lambda2, rho2, s, a):
    N = len(inputs) - 1
    temp = s[N] - tanh(a[N])
    lambda_new = (lambda2 - rho2 * temp) / 10
    return lambda_new


# update of lambda3
def update_lambda3(inputs, lambda3, rho3, o, v, s, c):
    N = len(inputs) - 1
    temp = o[N] - torch.mm(v, s[N]) - c
    lambda_new = (lambda3 - rho3 * temp) / 10
    return lambda_new
