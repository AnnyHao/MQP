import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data

# Define data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x < 0.7).float())  # Binarization
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

# Initialize1 C-TF-IDF matrices
pixels_per_image = images.shape[2] * images.shape[3]
num_classes = 10
c_tf_idf = np.zeros((num_classes, pixels_per_image))

# Calculate TF and IDF
tf = np.zeros((num_classes, pixels_per_image))
idf = np.zeros(pixels_per_image)

for i in range(num_classes):
    class_mask = labels == i
    class_images = images[class_mask].reshape(-1, pixels_per_image)
    tf[i, :] = np.mean(class_images, axis=0)

idf = np.mean(images.reshape(-1, pixels_per_image), axis=0)

# Calculate C-TF-IDF
# Calculate C-TF-IDF
A = np.mean(np.sum(images, axis=(1, 2, 3)))

# Adding a small constant to avoid division by zero
epsilon = 1e-10
idf += epsilon  # Ensure idf is never zero by adding a small constant

for i in range(num_classes):
    c_tf_idf[i, :] = tf[i, :] * np.log(1 + A / (idf + epsilon))

# Set the threshold
threshold = 0.1

# Apply thresholds, setting pixels below the threshold to 0 (black)
filtered_images = np.zeros_like(images)
for i in range(num_classes):
    class_mask = labels == i
    class_images = images[class_mask]
    filter_mask = c_tf_idf[i, :] > threshold
    for img in class_images:
        filtered_img = img.reshape(-1)
        filtered_img[filter_mask] = 1  # Set pixels above the threshold to white
        filtered_img[~filter_mask] = 0  # Set pixels below the threshold to black
        filtered_images[class_mask] = class_images.reshape(class_images.shape)

# Visualization
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flatten()):
    img = filtered_images[i].reshape(28, 28)
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.set_title(f'Digit: {i}')
    ax.axis('off')
plt.show()

