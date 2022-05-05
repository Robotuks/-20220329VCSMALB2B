
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data
Y = digits.target

pca = PCA(n_components=3)
pca.fit(X)
x_pca = pca.transform(X)
print(x_pca.shape)


fig = plt.figure()
ax = Axes3D(fig)
n_unique = np.unique(Y)
for unique_class in n_unique:
    mask = Y == unique_class
    ax.scatter(x_pca[mask, 0], x_pca[mask, 1], x_pca[mask, 2], cmap="Set1", label=unique_class)
ax.legend()
plt.show()