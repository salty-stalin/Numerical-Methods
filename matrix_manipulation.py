
import numpy
from matplotlib import pyplot
import scipy.integrate
import scipy.optimize

v = (1.0 + 1e-4 * numpy.random.rand(8)) #random.rand generate matrix between 0-1 with specified shape
print(v)

A = numpy.diagflat(v) #diagflat creates a 2d array with inputs as a diagonal
B = numpy.diagflat(v, -1)
print(A)
print(B)

A1 = numpy.reshape(v, (2, 4)) #reshape as a 2x4 matrix
A2 = numpy.reshape(v, (4, 2))#reshape as 4x2 matrix
A3 = numpy.reshape(v, (2, 2, 2))#a three dimensional 2 × 2 × 2 array.

print(A1)
print(A2)
print(A3)
