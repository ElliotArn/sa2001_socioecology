import numpy as np
import scipy
import matplotlib.pyplot as plt

# Parameters
k_a0 = 0.7
k_p0 = 1.2
k_q0 = 4
k_w0 = 0.2
d_a = 0.5
d_p = 0.2
d_w = 1
r = 1
r_q = 1
r_w = 1
Q = 1
t0 = 0
t_max = 1



# Derivative functions

def s(k_a):
    s1 = 0.1
    s2 = 1
    s3 = 0
    return s1/(1+np.e**(-s2*k_a+s3))

def f(k_a, k_p, k_w, k_q):
    A = 10
    alpha_a = 0.3
    alpha_p = 0.3
    alpha_w = 0.2
    alpha_q = 0.2
    return A * k_a**alpha_a * k_p**alpha_p * k_w**alpha_w * k_q**alpha_q

def fk_a(k_a, k_p, k_w, k_q):
    return s(k_a)*f(k_a, k_p, k_w, k_q) - (d_a+r)*k_a

def fk_p(k_p):
    return -d_w*k_p

def fk_w(k_w):
    return r_w - d_w*k_w

def fk_q(k_q):
    return r_q*k_q*(1-k_q/Q)


# System

def vfunc(t, k):
    k_a, k_p, k_w, k_q = k[0], k[1], k[2], k[3]
    return np.array([fk_a(k_a, k_p, k_w, k_q), fk_p(k_p), fk_w(k_w), fk_q(k_q)])

k0 = np.array([k_a0, k_p0, k_w0, k_q0])
t_span = [t0, t_max]

result = scipy.integrate.solve_ivp(vfunc, t_span, k0, max_step=0.1)

k = result.y

k_a, k_p, k_w, k_q = k[0], k[1], k[2], k[3]
time = result.t

plt.plot(time, k_a)
plt.show()