import torch
import torchvision
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Function to plot an image
def plot_image(image, title=""):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to apply t-SNE embedding to the data
def apply_tsne_embedding(trainset, testset):
    # Concatenate training and testing data to apply t-SNE on the entire dataset
    full_data = torch.cat((trainset.data, testset.data), dim=0)
    full_labels = torch.cat((trainset.targets, testset.targets), dim=0)

    # Select an image to display before and after transformation (just for visualization)
    img_idx = 0  # Index of the image to display (you can modify this for other images)
    original_image = full_data[img_idx].numpy()

    # Plot the original image
    plot_image(original_image, "Original Image")

    # Flatten images and randomize the features
    flattened_data = full_data.view(full_data.size(0), -1).numpy()
    randomized_data = flattened_data[:, np.random.permutation(flattened_data.shape[1])]
    
    # Randomized image
    randomized_image = randomized_data[img_idx].reshape(28, 28)
    
    # Plot the randomized image
    plot_image(randomized_image, "Randomized Image")

    # Transpose to treat features as "samples" in t-SNE
    transposed_data = randomized_data.T

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=40, metric="euclidean", max_iter=300, random_state=0)
    tsne_coordinates = tsne.fit_transform(transposed_data)

    # Normalize the t-SNE coordinates to fit within a 28x28 grid
    scaler = MinMaxScaler(feature_range=(0, 27))
    tsne_coordinates = scaler.fit_transform(tsne_coordinates).astype(int)

    # Initialize an array to hold the reorganized feature map
    reordered_data = np.zeros((full_data.size(0), 1, 28, 28))
    count_matrix = np.zeros((28, 28))

    # Map t-SNE coordinates to the 28x28 grid for each image
    for img_idx in range(full_data.size(0)):
        feature_vector = transposed_data[:, img_idx]
        grid = np.zeros((28, 28))
        count = np.zeros((28, 28))

        for feature_idx, (x, y) in enumerate(tsne_coordinates):
            # Accumulate feature values in the grid and count occurrences
            grid[x, y] += feature_vector[feature_idx]
            count[x, y] += 1

        # Average the values where multiple features overlap
        grid = np.divide(grid, count, out=np.zeros_like(grid), where=count != 0)
        reordered_data[img_idx, 0] = grid  # Place the averaged grid in the final dataset array

    # After the t-SNE transformation and feature reorganization, plot the transformed image
    transformed_image = reordered_data[img_idx, 0].numpy()  # Grab the transformed image

    # Plot the transformed image
    plot_image(transformed_image, "Transformed Image (After t-SNE)")

# Run the function to apply t-SNE and display the images
apply_tsne_embedding(trainset, testset)
