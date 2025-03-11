
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
X ,y = make_circles(100, factor=0.1, noise=.1)
plt.scatter(X[: ,0] ,X[: ,1] ,c=y ,s=50 ,cmap="rainbow")
plt.show()
def plot_svc_decision_function(model ,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0] ,xlim[1] ,30)
    y = np.linspace(ylim[0] ,ylim[1] ,30)
    Y ,X = np.meshgrid(y ,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P ,colors="k" ,levels=[-1 ,0 ,1] ,alpha=0.5 ,linestyles=["--" ,"-" ,"--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
clf = SVC(kernel = "linear").fit(X ,y)
plt.scatter(X[: ,0] ,X[: ,1] ,c=y ,s=50 ,cmap="rainbow")
plot_svc_decision_function(clf)
# r = np.exp(-( X**2).sum(1))
# rlim = np.linspace(min(r) ,max(r) ,100)
plt.show()

