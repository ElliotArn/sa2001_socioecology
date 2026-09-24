import numpy as np
import scipy
import matplotlib.pyplot as plt

# Parameters
k_a0 = 1
k_p0 = 1
k_q0 = 1
k_w0 = 1
d_a = 1
d_p = 1
d_w = 1
r = 1
r_q = 1
r_w = 1
Q = 1



# Derivative functions

def s(k_a):
    s1 = 1
    s2 = 1
    s3 = 1
    return s1/(1+np.e**(-s2*k_a+s3))

def f(k_a, k_p, k_w, k_q):
    A = 1
    alpha_a = 0.25
    alpha_p = 0.25
    alpha_w = 0.25
    alpha_q = 0.25
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

def vfunc(k_a, k_p, k_w, k_q):
    return np.array([fk_a(k_a, k_p, k_w, k_q), fk_p(k_p), fk_w(k_w), fk_q(k_q)])

k0 = np.array([k_a0, k_p0, k_w0, k_q0])
t = np.linspace(100)

k = scipy.integrate.odeint(vfunc, k0, t)

plt.plot(t, k)
plt.show()