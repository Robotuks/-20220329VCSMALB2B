import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


iris = datasets.load_iris()
X = iris.data
Y = iris.target

X_reduced = PCA(n_components=3).fit_transform(X)
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=Y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40
)
plt.show()