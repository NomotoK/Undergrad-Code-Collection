import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap



score = [ 0.42857, 0.54285, 0.45714,0.57142,0.60000, 0.65714,0.60201,0.71428, 0.65714,0.67431,0.68431,0.67922,0.65431,0.66000,0.67132,
          0.64112, 0.62389, 0.60000,0.63321,0.60021, 0.65320,0.63139,0.61239,0.60921, 0.62201,0.60019,0.61021,0.61932,0.59021,0.62123]
plt.figure()
plt.xlabel('values of K in KNN')
plt.ylabel('cross validation accuracy')
x1 = np.arange(0,30)
plt.plot(x1,score)
plt.show()
