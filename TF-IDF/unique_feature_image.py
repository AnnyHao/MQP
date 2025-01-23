import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Define data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x > 0.5).float())  # Binarization
])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

# Extract all data
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Convert to Numpy array for calculation
images = images.numpy()
labels = labels.numpy()

# Initialize TF and IDF matrices
pixels_per_image = images.shape[2] * images.shape[3]
tf = np.zeros((10, pixels_per_image))  # Frequency of pixels for each digit
idf = np.zeros(pixels_per_image)       # Global pixel frequency

# Calculate TF and IDF
for i in range(10):
    digit_masks = labels == i
    digit_images = images[digit_masks]
    # Calculate TF for each digit
    tf[i, :] = np.mean(digit_images.reshape(-1, pixels_per_image), axis=0)
# Calculate IDF
idf = np.mean(images.reshape(-1, pixels_per_image), axis=0)

# Calculate TF-IDF
tf_idf = np.zeros((10, pixels_per_image))
N = 10  # Total number of digits
for i in range(10):
    tf_idf[i, :] = tf[i, :] * np.log(N / (idf + 1e-10))  # Add a small constant to prevent division by zero

# Set the threshold to 50% of each digit's maximum TF-IDF value
thresholds = np.max(tf_idf, axis=1) * 0.5

# Apply thresholds, setting pixels below the threshold to 0
filtered_images = np.zeros_like(tf_idf)
for i in range(10):
    mask = tf_idf[i] >= thresholds[i]
    filtered_images[i, mask] = tf_idf[i, mask]
# Apply threshold, setting pixels below the threshold to 0 (black), and above the threshold to 1 (white)
binary_images = np.zeros_like(tf_idf)
for i in range(10):
    mask = tf_idf[i] >= thresholds[i]
    binary_images[i, mask] = 1  # Set pixels above the threshold to white

# Visualization
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    img = binary_images[i].reshape(28, 28)
    ax.imshow(img, cmap='gray', interpolation='nearest')  # Display in grayscale to show black and white images
    ax.set_title(f'Digit: {i}')
    ax.axis('off')
plt.show()
