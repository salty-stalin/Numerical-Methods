import numpy
import scipy.integrate
from matplotlib import pyplot

# Create an anonymous function defining the IVP
#y'(x) = −y(x), y(0) = 1, x ∈ [0, 10].
f = lambda y, x: -y
x = numpy.linspace(0, 10)



# Solve using odeint
solivp = scipy.integrate.odeint(f, [1.0], x) #(function, initial value, points of soltion)


pyplot.plot(x, solivp, 'r--', label="ODEint")
pyplot.plot(x, numpy.exp(-x), 'k-', label="Exact")
pyplot.xlabel(r"$x$")
pyplot.ylabel(r"$y$")
pyplot.legend();
pyplot.show()

solivp_accurate = scipy.integrate.odeint(f, [1.0], x, rtol = 1e-12, atol = 1e-12) #relaive and absolute tolerance
pyplot.semilogy(x, solivp, 'r--', label="ODEint") #plot log y
pyplot.semilogy(x, solivp_accurate, 'b:', label=r"ODEint (tolerance $10^{-12})$")
pyplot.semilogy(x, numpy.exp(-x), 'k-', label="Exact")
pyplot.xlabel(r"$x$")
pyplot.ylabel(r"$y$")
pyplot.legend();
pyplot.show()


#Solve the differential equation y'(x) = − cos(y(x)), y(0) = 1, x ∈ [0, 10]
def f1(solivp,x):
    dydx= -numpy.cos(solivp)
    return dydx
solve = scipy.integrate.odeint(f1, [1.0], x)

pyplot.plot(x, solve, 'r--', label="ODEint")
pyplot.xlabel(r"$x$")
pyplot.ylabel(r"$y$")
pyplot.legend();
pyplot.show()

#Solve the differential equation y'(x) = −C(x)y(x), y(0) = 1, x ∈ [0, 10].
#Define C as C = 1 + integral(0 to x) of sin(s)^2

def integrand(s):
    return numpy.sin(s)**2
def C(x):
    return 1.0 + scipy.integrate.quad(integrand, 0, x) [0]
def g(y, x):
    return -C(x) * y
solve1 = scipy.integrate.odeint(g, [1.0], x)
pyplot.plot(x, solve1)
pyplot.show()


#Solve the system x'(t) = -y(t) and y(t) = x(t) t ∈ [0, 500] x(0) = 1 y(0) =0

def system(z, t): #create a simultanous system using arrays
    dzdt = numpy.zeros_like(z)#Return an array of zeros with the same shape and type as a given array
    x = z[0]
    y = z[1]
    dzdt[0] = -y
    dzdt[1] = x
    return dzdt
z0 = [1.0, 0.0]
t = numpy.linspace(0, 500, 1000)
z = scipy.integrate.odeint(system, z0, t, atol = 1e-10, rtol = 1e-10)

x = z[:,0]
y = z[:,1]
r = numpy.sqrt(x**2 + y**2)
fig = pyplot.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131) #1 by 3 position 1
ax1.plot(t, x, label = r"$x$")
ax1.plot(t, y, label = r"$y$")
pyplot.legend()
ax1.set_xlabel(r"$t$")
6
ax2 = fig.add_subplot(132)
ax2.plot(x, y)
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$y$")
ax3 = fig.add_subplot(133)
ax3.plot(t, r)
ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$r$")
pyplot.show()

# Solving non-linear equations cos(x)-x = 0
def f2(x):
    dydx= numpy.cos(x) - x
    return dydx
s = scipy.optimize.brentq(f2,0.0,1.0)
print(s)