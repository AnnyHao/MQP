import numpy as np
import matplotlib.pyplot as plt
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

# Calculate entropy for an image
def calculate_entropy(images, is_color=False):
    entropies = []
    for img in images:
        if is_color:
            # Flatten across all channels
            img = img.reshape(-1, img.shape[-1])
        else:
            img = img.flatten()
        # Calculate histogram of pixel intensities
        hist, _ = np.histogram(img, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]  # Avoid log(0)
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist))
        entropies.append(entropy)
    return np.array(entropies)

# Plot histograms
def plot_entropy_histograms(mnist_data, k49_data, cifar10_data, bins=50):
    plt.figure(figsize=(12, 6))
    datasets = [
        ("MNIST", mnist_data),
        ("K-49", k49_data),
        ("CIFAR-10", cifar10_data),
    ]
    for i, (name, data) in enumerate(datasets, 1):
        plt.subplot(1, 3, i)
        plt.hist(data, bins=bins, alpha=0.7, color="purple", edgecolor="black")
        plt.title(f"{name} Entropy")
        plt.xlabel("Entropy")
        plt.xlim(0, 8)
        plt.ylabel("Frequency")
        plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.show()

# Load datasets
mnist_images = load_mnist()
k49_images = load_k49()
cifar10_images = load_cifar10()

# Compute entropy
mnist_entropy = calculate_entropy(mnist_images)
k49_entropy = calculate_entropy(k49_images)
cifar10_entropy = calculate_entropy(cifar10_images, is_color=True)

# Plot histograms
plot_entropy_histograms(mnist_entropy, k49_entropy, cifar10_entropy)
