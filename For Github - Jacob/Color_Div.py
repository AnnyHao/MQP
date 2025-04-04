import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import cv2

# Load datasets
def load_mnist():
    # MNIST data (grayscale)
    from keras.datasets import mnist
    (x_train, _), _ = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    return x_train

def load_k49():
    # K-49 data (grayscale)
    k49_train_imgs = np.load("k49-train-imgs.npz")['arr_0']
    k49_train_imgs = np.expand_dims(k49_train_imgs, axis=-1)  # Add channel dimension
    return k49_train_imgs

def load_cifar10():
    # CIFAR-10 data (color)
    (x_train, _), _ = cifar10.load_data()
    return x_train

# Calculate unique colors per image
def calculate_color_diversity(images):
    color_diversities = []
    for img in images:
        unique_colors = np.unique(img.reshape(-1, img.shape[-1]), axis=0)
        color_diversities.append(len(unique_colors))
    return np.array(color_diversities)

# Plot histograms
def plot_histograms(mnist_data, k49_data, cifar10_data, bins=50):
    plt.figure(figsize=(12, 6))
    datasets = [
        ("MNIST", mnist_data),
        ("K-49", k49_data),
        ("CIFAR-10", cifar10_data),
    ]
    for i, (name, data) in enumerate(datasets, 1):
        plt.subplot(1, 3, i)
        plt.hist(data, bins=bins, alpha=0.7, color="blue", edgecolor="black")
        plt.title(f"{name} Color Diversity")
        plt.xlabel("Unique Colors (Normalized)")
        plt.xlim(0, 1)
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.show()

# Load and preprocess datasets
mnist_images = load_mnist()
k49_images = load_k49()
cifar10_images = load_cifar10()

# Compute color diversity
mnist_diversity = calculate_color_diversity(mnist_images) / (mnist_images.shape[1] * mnist_images.shape[2])
k49_diversity = calculate_color_diversity(k49_images) / (k49_images.shape[1] * k49_images.shape[2])
cifar10_diversity = calculate_color_diversity(cifar10_images) / (cifar10_images.shape[1] * cifar10_images.shape[2])

# Plot histograms
plot_histograms(mnist_diversity, k49_diversity, cifar10_diversity)
