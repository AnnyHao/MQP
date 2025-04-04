import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10

# Load datasets
def load_mnist():
    from keras.datasets import mnist
    (x_train, _), _ = mnist.load_data()
    return x_train

def load_k49():
    k49_train_imgs = np.load("k49-train-imgs.npz")['arr_0']
    return k49_train_imgs

def load_cifar10():
    (x_train, _), _ = cifar10.load_data()
    return x_train

# Apply Sobel edge detection and calculate edge density
def calculate_edge_density(images, is_color=False):
    densities = []
    for img in images:
        if is_color:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        edges_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(edges_x**2 + edges_y**2)  # Combine X and Y gradients
        edge_pixels = np.sum(edges > 50)  # Threshold to detect edges
        total_pixels = img.shape[0] * img.shape[1]
        densities.append(edge_pixels / total_pixels)
    return np.array(densities)

# Plot histograms
def plot_edge_density_histograms(mnist_data, k49_data, cifar10_data, bins=50):
    plt.figure(figsize=(12, 6))
    datasets = [
        ("MNIST", mnist_data),
        ("K-49", k49_data),
        ("CIFAR-10", cifar10_data),
    ]
    for i, (name, data) in enumerate(datasets, 1):
        plt.subplot(1, 3, i)
        plt.hist(data, bins=bins, alpha=0.7, color="green", edgecolor="black")
        plt.title(f"{name} Edge Density")
        plt.xlabel("Edge Density")
        plt.xlim(0, 1)
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.show()

# Load datasets
mnist_images = load_mnist()
k49_images = load_k49()
cifar10_images = load_cifar10()

# Compute edge densities
mnist_edge_density = calculate_edge_density(mnist_images)
k49_edge_density = calculate_edge_density(k49_images)
cifar10_edge_density = calculate_edge_density(cifar10_images, is_color=True)

# Plot histograms
plot_edge_density_histograms(mnist_edge_density, k49_edge_density, cifar10_edge_density)
