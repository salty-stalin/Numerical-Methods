import pytest

import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import scipy
from scipy import integrate
from scipy import optimize

def question_1():
    """
    Solution to question 1 goes here
    """
    A = numpy.array([[2.0, 7.0], [3.0, 6.0]])
    return numpy.linalg.inv(A) 

def question_2():
    """
    Solution to question 2 goes here
    """
    v = numpy.linspace(1.0, 3.0, 20)
    return numpy.linalg.norm(v, ord=None, axis=None, keepdims=False)
    

def question_3():
    """
    Solution to question 3 goes here
    """
    v = numpy.linspace(1.0, 3.0, 20)
    y = numpy.exp(-v) * numpy.sin(2*numpy.pi * v)
    return numpy.sum(y[1::2])

def question_4():
    """
    Solution to question 4 goes here
    """
    C = [[12.0, 3.0, -2.0], [6.0, 4.0, 1.0], [0.0, 9.0, 2.0]]
    b = [1.0, 7.0, 3.0]
    return numpy.linalg.solve(C, b)

def question_5():
    """
    Solution to question 5 goes here
    """
    C = [[12.0, 3.0, -2.0], [6.0, 4.0, 1.0], [0.0, 9.0, 2.0]]
    evals, evecs = numpy.linalg.eig(C)
    return numpy.min(numpy.abs(evals))

def question_6():
    """
    Solution to question 6 goes here
    """
    A = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    return A

def question_7():
    """
    Solution to question 7 goes here
    """
    x = numpy.linspace(0.0, 5.0, 500)
    y = numpy.exp(-x) * numpy.sin(2*numpy.pi * x)
    pyplot.plot(x, y)
    pyplot.xlabel(r"$x$")
    pyplot.ylabel(r"$exp(-x)*sin(2*pi*x)$")
    pyplot.title("Figure for question 7")
    pyplot.show()

def question_8():
    """
    Solution to question 8 goes here
    """
    x = numpy.linspace(0.0, 1.0, 40)
    y = numpy.linspace(0.0, 2.0, 40)
    X, Y = numpy.meshgrid(x, y)
    Z = numpy.exp(-(X**2+Y**2))*numpy.cos(2.0*numpy.pi*X**2)*numpy.sin(4*numpy.pi*Y)
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = 'coolwarm')
    pyplot.show()

def question_9():
    """
    Solution to question 9 goes here
    """
    def f(x):
        return (numpy.exp(-x)*numpy.sin(x))/(1+x**2+0.1*x**4)

    I,err=scipy.integrate.quad(f, 0, 1) #integrate function , lower lim. , upper lim.
    return I    

def question_10():
    """
    Solution to question 10 goes here
    """
    y0 = [0, 0.5]
    y = numpy.zeros((100, 2))
    y[0, :] = y0
    for i in range(99):
        u, v = y[i, :]
        y[i+1, :] = [numpy.tan(v)**3, numpy.cos(u)**2]
    return y[9, :], y[99, :]

if __name__ == "__main__":
    pytest.main()
    question_7()
    question_8()
    