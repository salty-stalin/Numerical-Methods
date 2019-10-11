import numpy 
from matplotlib import pyplot
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm


#Plotting simple functions

x = numpy.linspace(0.0, 1.0, 80)
y = numpy.linspace(0.0, 2.0, 60)
sin = numpy.sin(x)
e = numpy.exp(-y**2)
pyplot.plot(x, sin)


pyplot.show()

pyplot.plot(y, e, 'r--', linewidth = 2, label = 'A simple function')
pyplot.xlabel(r"$y$")
pyplot.ylabel(r"$e^{-y^2}$")
pyplot.title("A simple plot")
pyplot.legend()

pyplot.show()

pyplot.plot(x, sin, marker = "o", markersize = 8)
pyplot.xlabel("x", size=16)

pyplot.show()

#Plotting several subplots
fig = pyplot.figure(figsize = (12, 4))
ax1 = fig.add_subplot(131)
ax1.semilogx(x,sin)
ax2 = fig.add_subplot(132)
ax2.semilogy(x,sin)
ax3 = fig.add_subplot(133)
ax3.loglog(x,sin)
fig.tight_layout()

pyplot.show()

#3D Plots
X, Y = numpy.meshgrid(x, y)

"""We could not build z directly from the vectors x, y as they have different sizes. The variables
X, Y are instead matrices with the same size which can be used"""

Z = numpy.cos(2.0 * numpy.pi * X * Y**2)
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

pyplot.show()
#3D Wireframe plot
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z)
ax.view_init(30, 45)
pyplot.show()
#3D Surface plot
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap = cm.coolwarm)
pyplot.show()

#X-Y Unit circle plot
theta = numpy.linspace(0, 2.0 * numpy.pi)
x = numpy.cos(theta)
y = numpy.sin(theta)
pyplot.plot(theta, x, label=r"$x$")
pyplot.plot(theta, y, label=r"$y$")
pyplot.xlabel(r"$x$")
pyplot.legend()
pyplot.show()

#3D Unit circle plot
fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, 0);
pyplot.show()

#Matrix effects on unit circle
A = numpy.random.randn(2,2)
print(numpy.linalg.eig(A))

x1 = x.copy()
y1 = y.copy()
r = numpy.vstack((x, y))
for i in range(len(x)):
    r1 = numpy.dot(A, r[:,i])
    x1[i] = r1[0]
    y1[i] = r1[1]

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, 0, label = "Original")
ax.plot3D(x1, y1, 0, label = r"Circle $\times A$")
pyplot.legend();
pyplot.show()

x2 = x.copy()
y2 = y.copy()
x5 = x.copy()
y5 = y.copy()
r = numpy.vstack((x, y))

for i in range(len(x)):
    r2 = numpy.dot(numpy.linalg.matrix_power(A,2), r[:,i]) #A^2 Effect
    r5 = numpy.dot(numpy.linalg.matrix_power(A,5), r[:,i]) #A^5 Effect
    x2[i] = r2[0]
    y2[i] = r2[1]
    x5[i] = r5[0]
    y5[i] = r5[1]

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(x, y, 0, label = "Original")
ax.plot3D(x1, y1, 0, label = r"Circle $\times A$")
ax.plot3D(x2, y2, 0, label = r"Circle $\times A^2$")
ax.plot3D(x5, y5, 0, label = r"Circle $\times A^5$")
pyplot.legend();
pyplot.show()

"""The circles become ellipses. The semi-major axes align themselves with the eigenvectors. The
size of the axis is multiplied by the appropriate eigenvalue each time: eigenvalues greater than
one lead to the size diverging."""
