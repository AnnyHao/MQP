import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    images = mnist.data.values.reshape(-1, 28, 28).astype(np.uint8)
    labels = mnist.target.astype(int)
    return images, labels

def image_to_binary(image_array):
    return (image_array < 128).astype(int)

def binary_to_decimal(binary_image, block_size):
    h, w = binary_image.shape
    n_h = h // block_size
    n_w = w // block_size
    decimal_matrix = np.zeros((n_h, n_w), dtype=int)
    for i in range(n_h):
        for j in range(n_w):
            block = binary_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            binary_str = ''.join(map(str, block.flatten()))
            decimal_matrix[i, j] = int(binary_str, 2)
    return decimal_matrix

def convert_images(images, block_size):
    return np.array([binary_to_decimal(image_to_binary(img), block_size) for img in images])

images, labels = load_mnist_data()
block_size = 4
images = convert_images(images, block_size)
labels = torch.tensor(labels, dtype=torch.long)

images = torch.tensor(images, dtype=torch.float32) / 255.0

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class UniqueFeatureClassifier(nn.Module):
    def __init__(self):
        super(UniqueFeatureClassifier, self).__init__()
        self.class_features = {}

    def fit(self, x_train, y_train):
        for i in range(10):
            indices = (y_train == i).nonzero(as_tuple=True)[0]
            class_images = x_train[indices]
            self.class_features[i] = class_images.mean(dim=0)

    def forward(self, x):
        similarities = torch.stack([torch.norm(self.class_features[i] - x, dim=(1, 2)) for i in range(10)]).T
        predictions = torch.argmin(similarities, dim=1)
        return predictions

model = UniqueFeatureClassifier()
model.fit(x_train, y_train)

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            predictions = model(data)
            correct += (predictions == target).sum().item()
            total += target.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

test(model, test_loader)
