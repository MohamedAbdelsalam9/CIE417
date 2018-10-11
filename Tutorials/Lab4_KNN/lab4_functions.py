import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_iris_data(features, labels):
    f, ax = plt.subplots(1, 1)
    ax.scatter(features[labels == 0, 0], features[labels == 0, 1], color="#d00000", label="Iris Setosa")
    ax.scatter(features[labels == 1, 0], features[labels == 1, 1], color="#0000d0", label="Iris Versicolour")
    ax.scatter(features[labels == 2, 0], features[labels == 2, 1], color="#00d000", label="Iris Virginica")
    ax.set_xlabel("Sepal Length (standardized)")
    ax.set_ylabel("Sepal Width (standardized)")
    ax.legend()
    f.set_size_inches(16, 6)
    ax.set_title("Iris Plant Type (training set)")
    plt.show()
    

def plot_knn_decision_boundaries(classifier, X, y, mesh_step):
    x1_limits = [min(X[:,0])-1, max(X[:,0])+1]
    x2_limits = [min(X[:,1])-1, max(X[:,1])+1]
    xx1, xx2 = np.meshgrid(np.arange(x1_limits[0], x1_limits[1], mesh_step), np.arange(x2_limits[0], x2_limits[1], mesh_step))
    mesh_points = np.stack((xx1.ravel(), xx2.ravel()), axis=1)
    yy = classifier.predict(mesh_points)
    yy = yy.reshape(xx1.shape)
    
    f, ax = plt.subplots(1, 1)
    cmap_points = ListedColormap(["#d00000", "#0000d0", "#00d000"])
    cmap_mesh = ListedColormap(["#ffaaaa", "#aaaaff", "#aaffaa"])

    if ((yy == 0).all()):
        ax.set_facecolor(cmap_mesh.colors[0])
    elif ((yy == 1).all()):
        ax.set_facecolor(cmap_mesh.colors[1])
    elif ((yy == 2).all()):
        ax.set_facecolor(cmap_mesh.colors[2])
    else:
        ax.pcolormesh(xx1, xx2, yy, cmap=cmap_mesh)
    
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=cmap_points.colors[0], label="Iris Setosa")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=cmap_points.colors[1], label="Iris Versicolour")
    ax.scatter(X[y == 2, 0], X[y == 2, 1], color=cmap_points.colors[2], label="Iris Virginica")
    ax.set_xlabel("Sepal Length (standardized)")
    ax.set_ylabel("Sepal Width (standardized)")
    ax.legend()
    f.set_size_inches(16, 6)
    ax.set_title(f"Iris Plant Type (training set), k = {classifier.n_neighbors}, weights = {classifier.weights}")
    plt.show()
    
    
def plot_knn_decision_boundaries_(classifier, X, y, mesh_step):
    x1_limits = [min(X[:,0])-0.25, max(X[:,0])+0.25]
    x2_limits = [min(X[:,1])-0.25, max(X[:,1])+0.25]
    xx1, xx2 = np.meshgrid(np.arange(x1_limits[0], x1_limits[1], mesh_step), np.arange(x2_limits[0], x2_limits[1], mesh_step))
    mesh_points = np.stack((xx1.ravel(), xx2.ravel()), axis=1)
    yy = classifier.predict(mesh_points)
    yy = yy.reshape(xx1.shape)
    
    f, ax = plt.subplots(1, 1)
    cmap_points = ListedColormap(["#0000d0", "#d00000"])
    cmap_mesh = ListedColormap(["#aaaaff", "#ffaaaa"])

    if ((yy == 0).all()):
        ax.set_facecolor(cmap_mesh.colors[0])
    elif ((yy == 1).all()):
        ax.set_facecolor(cmap_mesh.colors[1])
    else:
        ax.pcolormesh(xx1, xx2, yy, cmap=cmap_mesh)
    
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=cmap_points.colors[0])
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=cmap_points.colors[1])
    f.set_size_inches(16, 6)
    ax.set_title(f"k = {classifier.n_neighbors}, weights = {classifier.weights}")
    plt.show()