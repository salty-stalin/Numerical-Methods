import numpy as np
import scipy.integrate
import scipy.optimize

def f1(x):
    return np.sin(x)**2

y=scipy.integrate.quad(f1, 0, np.pi) #integrate function , lower lim. , upper lim.
print(y)

def f3(x):
    return np.exp(-x**2 * np.cos(2.0 * np.pi * x)**2)
q3 = scipy.integrate.quad(f3, 0, 1)
print(q3)

q3_accurate = scipy.integrate.quad(f3, 0, 1, epsabs = 1e-14, epsrel = 1e-14) #including absolute error tolerance and relative error tolerance
print(q3_accurate)