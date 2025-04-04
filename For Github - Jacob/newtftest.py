import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Precompute TF-IDF weights
def compute_tfidf_weights(dataset):
    data = dataset.data.float() / 255.0  # Normalize pixel intensities
    data = data.view(len(data), -1)  # Flatten images

    # Term Frequency (TF)
    tf = data.mean(dim=0)

    # Document Frequency (DF)
    df = (data > 0).sum(dim=0)

    # Inverse Document Frequency (IDF)
    num_images = len(data)
    idf = torch.log((num_images / (df + 1)))

    # Compute TF-IDF
    tf_idf = tf * idf

    # Normalize TF-IDF to [0, 1]
    tf_idf_normalized = (tf_idf - tf_idf.min()) / (tf_idf.max() - tf_idf.min())

    return tf_idf_normalized.view(-1)


tfidf_weights = compute_tfidf_weights(train_dataset)


# Define the TF-IDF weighted layer
class TFIDFWeightedLayer(nn.Module):
    def __init__(self, tfidf_weights):
        super().__init__()
        # Reshape TF-IDF weights to match image dimensions
        self.tfidf_weights = nn.Parameter(tfidf_weights.view(1, 1, 28, 28), requires_grad=False)

    def forward(self, x):
        return x * self.tfidf_weights


# Define the CNN with TF-IDF
class SimpleCNNWithTFIDF(nn.Module):
    def __init__(self, tfidf_weights):
        super().__init__()
        self.tfidf_layer = TFIDFWeightedLayer(tfidf_weights)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc = None

    def forward(self, x):
        x = self.tfidf_layer(x)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if self.fc is None:
            self.fc = nn.Linear(x.shape[1], 10).to(device)

        x = self.fc(x)
        return x


# Initialize model, optimizer, and loss function
model = SimpleCNNWithTFIDF(tfidf_weights).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# Training loop
def train(model, device, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        print(
            f'Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')


# Evaluation loop
def evalmodel(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {100. * correct / total:.2f}%')


# Run training and evaluation
train(model, device, train_loader, optimizer, criterion)
evalmodel(model, device, test_loader, criterion)
