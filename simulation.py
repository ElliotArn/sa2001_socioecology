import numpy as np
import scipy
import matplotlib.pyplot as plt

# Parameters
k_a0 = 3.7
k_p0 = 1.2
k_q0 = 2
k_w0 = 0.2
d_a = 0.5
d_p = 0.2
d_w = 1
r = 0.2
r_q = 1
r_q_min = 0.1
r_w = 1
Q = 2
t0 = 0
t_max = 39
p_p = 0.01



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

def fk_a_baseline(k_a, k_p, k_w, k_q):
    return s(k_a)*f(k_a, k_p, k_w, k_q) - (d_a+r)*k_a

def fk_p_baseline(k_p, k_a):
    return -d_p*k_p + p_p * k_a

def fk_w_baseline(k_w):
    return r_w - d_w*k_w

def fk_q_baseline(k_q, a_f):
    return r_q*k_q*(1-k_q/Q)

def fk_q_alt(k_q, a_f):
    if k_q < r_q_min:
        return 0
    return r_q*k_q*(1-k_q/Q) - max(a_f, 0)


# System
def test(fk_a=fk_a_baseline, fk_p=fk_p_baseline, fk_w=fk_w_baseline, fk_q=fk_q_baseline, title=""):
    def vfunc(t, k):
        k_a, k_p, k_w, k_q = k[0], k[1], k[2], k[3]
        return np.array([fk_a(k_a, k_p, k_w, k_q), fk_p(k_p, k_a), fk_w(k_w), fk_q(k_q, fk_p(k_p, k_a))])

    k0 = np.array([k_a0, k_p0, k_w0, k_q0])
    t_span = [t0, t_max]

    result = scipy.integrate.solve_ivp(vfunc, t_span, k0, max_step=0.1)

    k = result.y

    k_a, k_p, k_w, k_q = k[0], k[1], k[2], k[3]
    time = result.t
    make_plot(time, k_a, k_p, k_w, k_q, title)

def make_plot(time, k_a, k_p, k_w, k_q, title=""):
    plt.plot(time, k_a, label='Assets')
    plt.plot(time, k_p, label='Phosphorous')
    plt.plot(time, k_w, label='Water')
    plt.plot(time, k_q, label='Soil quality')
    plt.xlabel('time')
    plt.legend()
    plt.title(title)
    plt.show()


# baseline
test(title="Baseline")

# lägga till fosfor från staten
amount_p = 0.5
fk_p_alt = lambda k_p, k_a : -d_p*k_p + amount_p + p_p * k_a

test(fk_p=fk_p_alt, title="Add phosphorous")


# lägga till vatten från staten
#fk_p = lambda k_p, k_a : -d_p*k_p
amount_w = 1
fk_w_alt = lambda k_r :r_w- d_w*k_r + amount_w
test(fk_w=fk_w_alt, title="Add water")


# lägga till vatten och fosfor från staten
fk_p_alt = lambda k_p, k_a : -d_p*k_p + amount_p + p_p * k_a
test(fk_p=fk_p_alt, fk_w=fk_w_alt, title="Add water and phosphorous")
