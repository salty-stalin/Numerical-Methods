import matplotlib.pyplot as plt
import numpy as np
x =np.linspace(0, 1.0, num=80)
y = np.linspace(0, 2.0, num=80)

sin =np.sin(x)
exp =np.exp(y)
plt.plot(sin,exp)
plt.show()