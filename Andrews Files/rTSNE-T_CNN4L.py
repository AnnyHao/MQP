import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.manifold import TSNE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Function to apply PCA and t-SNE embedding to the data
def apply_tsne_embedding(trainset, testset):
    # Concatenate training and testing data to apply t-SNE on the entire dataset
    full_data = torch.cat((trainset.data, testset.data), dim=0)
    full_labels = torch.cat((trainset.targets, testset.targets), dim=0)

    # Flatten images and randomize the features
    flattened_data = full_data.view(full_data.size(0), -1).numpy()
    randomized_data = flattened_data[:, np.random.permutation(flattened_data.shape[1])]
    
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

    # Split back into training and testing sets
    train_data = torch.tensor(reordered_data[:len(trainset)], dtype=torch.float32)
    train_labels = trainset.targets
    test_data = torch.tensor(reordered_data[len(trainset):], dtype=torch.float32)
    test_labels = testset.targets

    return train_data, train_labels, test_data, test_labels

# Preprocess the datasets
train_data, train_labels, test_data, test_labels = apply_tsne_embedding(trainset, testset)

# Create dataloader with preprocessed data
trainloader = torch.utils.data.DataLoader(list(zip(train_data, train_labels)), batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(list(zip(test_data, test_labels)), batch_size=64, shuffle=False)

# Define the 4-layer CNN model
class FourLayerCNN(nn.Module):
    def __init__(self):
        super(FourLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
def initialize_model():
    model = FourLayerCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, criterion, optimizer

# Train the model
def train_model(model, trainloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}")

# Evaluate the model
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy


# Run the model multiple times and calculate the average accuracy
def run_multiple_trials(num_trials, epochs):
    accuracies = []
    for trial in range(num_trials):
        print(f"\n--- Trial {trial + 1} ---")
        model, criterion, optimizer = initialize_model()
        train_model(model, trainloader, criterion, optimizer, epochs)
        accuracy = evaluate_model(model, testloader)
        accuracies.append(accuracy)
    
    avg_accuracy = np.mean(accuracies)
    print(f"\nAverage Accuracy over {num_trials} trials: {avg_accuracy}%")
    return avg_accuracy

# Execute the trials
run_multiple_trials(5, 16)