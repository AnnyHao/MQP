import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class TFIDFWeightedLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.tfidf_weights = nn.Parameter(torch.ones(num_features), requires_grad=False)

    def forward(self, x):
        return x * self.tfidf_weights.view(1, -1, 1, 1)


class SimpleCNNWithTFIDF(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2)
        self.fc = None

    def forward(self, x):
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

model = SimpleCNNWithTFIDF().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

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

        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100. * correct / total:.2f}%')

def test(model, device, test_loader, criterion):
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

    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {100. * correct / total:.2f}%')

train(model, device, train_loader, optimizer, criterion)
test(model, device, test_loader, criterion)
