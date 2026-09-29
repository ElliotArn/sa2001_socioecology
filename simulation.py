import numpy as np
import scipy
import matplotlib.pyplot as plt


# Initial values (not varied between scenarios)
k_a0 = 3.7
k_p0 = 1.2
k_q0 = 2
k_w0 = 0.2
t0 = 0

# Other constants not varied between scenarios
t_max = 10
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


# Classes

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
        return 0 # we assume the system is independent on soil quality (not a limiting factor) so we can therefore let the soil quality be constant

class Agrochemicals:
    # Scenario 1
    # We assume good water conditions

    def __init__(self, strong):
        self.s1 = 0.25
        self.s2 = 2.5
        self.s3 = 20
        self.A = 10
        self.alpha_a = 0.4
        self.alpha_p = 0.3
        self.alpha_q = 0.2
        self.d_a = 0.7
        self.c1 = 1
        self.c2 = 20

        if strong:
            self.c3 = 4 # strong negative effect on soil quality
        else:
            self.c3 = 1 # mild negative effect on soil quality

        self.c4 = 4
        self.d_p = 0.2
        self.r_q = 1
        self.Q = 10

    def I_p(self, k_a):
        return self.c1*k_a**2/(self.c2+k_a**2)

    def I_q(self, k_a):
        return self.c3*k_a/(self.c4+k_a)

    def f(self, k_a, k_p, k_w, k_q):
        return self.A * k_a**self.alpha_a * k_p**self.alpha_p * k_q**self.alpha_q

    def fk_a(self, k_a, k_p, k_w, k_q):
        return s(k_a, self.s1, self.s2, self.s3)*self.f(k_a, k_p, k_w, k_q) - (self.d_a+r)*k_a

    def fk_p(self, k_a, k_p, k_w, k_q):
        return self.I_p(k_a) - self.d_p*k_p

    def fk_w(self, k_a, k_p, k_w, k_q): # It is assumed that the household invests in water management so that water is not a limiting factor for crop growth; therefore we omit water from the system. For simplicity we will assume no change in water over time.
        # So: water is not interesting in this scenario
        return 0

    def fk_q(self, k_a, k_p, k_w, k_q):
        return self.r_q*k_q*(1-k_q/self.Q)-self.I_q(k_a)*k_q

class Energy:
    # Scenario 2
    # We assume good water conditions
    # Because of diverse energy sources, manure can be used as fertiliser instead of fuel

    def __init__(self, sufficient):
        self.s1 = 0.1
        self.s2 = 1
        self.s3 = 0
        self.A = 10
        self.alpha_a = 0.3
        self.alpha_p = 0.3
        self.alpha_w = 0.2 # it says alpha_q in the article but I think they meant to write alpha_w
        self.d_a = 0.5

        if sufficient:
            self.c1 = 1 # sufficient amount of nutrient rich manure
        else:
            self.c1 = 0.5 # insufficient amount or nutrient poor manure

        self.c2 = 5
        self.c3 = 1
        self.c4 = 1.8
        self.d_p = 0.2
        self.r_w = 1
        self.c5 = 1
        self.c6 = 40
        self.d_w = 1

        # From baseline scenario:
        self.d_p = 1

    def I_p(self, k_a, k_p):
        return self.c1*k_a**2/(self.c2+k_a**2) * self.c3*k_p/(self.c4+k_p)

    def I_w(self, k_a):
        return self.c5*k_a**2/(self.c6+k_a**2)

    def f(self, k_a, k_p, k_w, k_q):
        return self.A * k_a**self.alpha_a * k_p**self.alpha_p * k_w**self.alpha_w

    def fk_a(self, k_a, k_p, k_w, k_q):
        return s(k_a, self.s1, self.s2, self.s3)*self.f(k_a, k_p, k_w, k_q) - (self.d_a+r)*k_a

    def fk_p(self, k_a, k_p, k_w, k_q):
        return self.I_p(k_a, k_p) - self.d_p*k_p # (assumption: manure improves nutrient level)
    
    def fk_w(self, k_a, k_p, k_w, k_q):
        return self.r_w + self.I_w(k_a)*k_w - self.d_w*k_w # (assumption: improved water conditions are enabled by rainwater harvest) (we cannot act like water is not a limiting factor as in scenario 1, because the household also needs to invest some money into other energy sources)
    
    def fk_q(self, k_a, k_p, k_w, k_q):
        # The system does not depend on soil quality so we will say it is constant
        return 0
    
class Tillage:
    # Scenario 3

    def __init__(self):
        self.s1 = 0.1
        self.s2 = 10
        self.s3 = 20
        self.A = 6
        self.alpha_a = 0.4
        self.alpha_w = 0.4
        self.d_a = 1
        self.r_w = 1
        self.d_w = 0.2

    def f(self, k_a, k_p, k_w, k_q):
        return self.A * k_a**self.alpha_a * k_w**self.alpha_w

    def fk_a(self, k_a, k_p, k_w, k_q):
        return s(k_a, self.s1, self.s2, self.s3)*self.f(k_a, k_p, k_w, k_q) - (self.d_a+r)*k_a

    def fk_p(self, k_a, k_p, k_w, k_q):
        return 0 # the conservation tillage makes the phosphorous loss zero

    def fk_w(self, k_a, k_p, k_w, k_q):
        return self.r_w - self.d_w*k_w

    def fk_q(self, k_a, k_p, k_w, k_q):
        # The system does not depend on soil quality so we will say it is constant
        return 0


# - Main -

# Scenario 0: Baseline
baseline = Baseline()
run(scenario=baseline, title="Scenario: Baseline")

# Scenario 1: Input of agrochemicals
# - 1a: Mild negative effect of agrochemicals on soil quality
mild_agro = Agrochemicals(False)
run(scenario=mild_agro, title="Scenario: Agrochemicals, mild")
# - 1b: Strong negative effect of agrochemicals on soil quality
strong_agro = Agrochemicals(True)
run(scenario=strong_agro, title="Scenario: Agrochemicals, strong")

# Scenario 2: Diversification of household energy sources
# - 2a: Sufficient amount of nutrient rich manure
energy_sufficient = Energy(True)
run(scenario=energy_sufficient, title="Scenario: Energy, sufficient")
# - 2b: Strong negative effect of agrochemicals on soil quality
energy_insufficient = Energy(False)
run(scenario=energy_insufficient, title="Scenario: Energy, insufficient")

# Scenario 3: Conservation tillage
tillage = Tillage()
run(scenario=tillage, title="Scenario: Tillage")

print("Done")


# Investigate:
# - Water preserving techniques
# - Regenerative agriculture