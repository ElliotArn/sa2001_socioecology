import numpy as np
import scipy
import matplotlib.pyplot as plt

class TestClass:

    def __init__(self):
        pass

    def test_method(self):
        print(1)


def test_func(fun):
    fun()

test_class = TestClass()
#test_func(test_class.test_method)


# Initial values
k_a0 = 3.7
k_p0 = 1.2
k_q0 = 2
k_w0 = 0.2
t0 = 0

# Constants not varied between scenarios
t_max = 39
r = 0.2


# Help functions (general)

def s(k_a, s1, s2, s3):
    return s1/(1+np.e**(-s2*k_a+s3))

def make_plot(time, k_a, k_p, k_w, k_q, title=""):
    plt.plot(time, k_a, label='Assets')
    plt.plot(time, k_p, label='Phosphorous')
    plt.plot(time, k_w, label='Water')
    plt.plot(time, k_q, label='Soil quality')
    plt.xlabel('time')
    plt.legend()
    plt.title(title)
    plt.show()

class Baseline:

    def __init__(self):
        self.s1 = 0.1
        self.s2 = 10
        self.s3 = 20
        self.A = 10
        self.alpha_a = 0.3
        self.alpha_p = 0.3
        self.alpha_w = 0.3
        self.d_a = 1
        self.d_p = 1
        self.r_w = 1.5
        self.d_w = 0.5

    def f(self, k_a, k_p, k_w, k_q):
        return self.A * k_a**self.alpha_a * k_p**self.alpha_p * k_w**self.alpha_w

    def fk_a(self, k_a, k_p, k_w, k_q):
        return s(k_a, self.s1, self.s2, self.s3)*self.f(k_a, k_p, k_w, k_q) - (self.d_a+r)*k_a

    def fk_p(self, k_a, k_p, k_w, k_q):
        return -self.d_p*k_p

    def fk_w(self, k_a, k_p, k_w, k_q):
        return self.r_w - self.d_w*k_w

    def fk_q(self, k_a, k_p, k_w, k_q):
        return 0


def run(scenario, title=""):
    def vfunc(t, k):
        k_a, k_p, k_w, k_q = k[0], k[1], k[2], k[3]
        return np.array([scenario.fk_a(k_a, k_p, k_w, k_q), scenario.fk_p(k_a, k_p, k_w, k_q), scenario.fk_w(k_a, k_p, k_w, k_q), scenario.fk_q(k_a, k_p, k_w, k_q)])

    k0 = np.array([k_a0, k_p0, k_w0, k_q0])
    t_span = [t0, t_max]

    result = scipy.integrate.solve_ivp(vfunc, t_span, k0, max_step=0.1)

    k = result.y

    k_a, k_p, k_w, k_q = k[0], k[1], k[2], k[3]
    time = result.t
    make_plot(time, k_a, k_p, k_w, k_q, title)



baseline = Baseline()
run(scenario=baseline, title="Baseline scenario")
print("Done")