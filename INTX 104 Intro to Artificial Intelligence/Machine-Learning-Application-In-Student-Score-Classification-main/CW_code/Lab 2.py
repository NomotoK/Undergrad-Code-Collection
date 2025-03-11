import numpy as np
import matplotlib.pyplot as plt

xdata = 7 * np.random.random(100)
ydata = np.sin(xdata) + 0.25 * np.random(100)
zdata = np.exp(xdata) + 0.25 * np.random(100)

fig = plt.figure(figsize=(9,6))

ax = plt.axes(projection = '3d')
ax.scatter3D(xdata,ydata,zdata)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()