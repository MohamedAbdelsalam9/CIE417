import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_iris_data(features, labels):
    f, ax = plt.subplots(1, 1)
    ax.scatter(features[labels == 0, 0], features[labels == 0, 1], color="#d00000", label="Iris Setosa")
    ax.scatter(features[labels == 1, 0], features[labels == 1, 1], color="#0000d0", label="Iris Versicolour")
    ax.scatter(features[labels == 2, 0], features[labels == 2, 1], color="#00d000", label="Iris Virginica")
    ax.set_xlabel("First Principal Component")
    ax.set_ylabel("Second Principal Component")
    ax.legend()
    f.set_size_inches(16, 6)
    ax.set_title("Iris Plant Type (training set)")
    plt.show()