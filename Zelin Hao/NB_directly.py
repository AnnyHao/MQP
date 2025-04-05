import torch
import torchvision
from torchvision import datasets, transforms
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB


transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=len(mnist_data), shuffle=True)

images, labels = next(iter(data_loader))
images = images.view(images.size(0), -1).numpy() * 255
labels = labels.numpy()

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)

# model = GaussianNB()
model = MultinomialNB()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')


# MultinomialNB ~82%
