from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torchvision import datasets, transforms

# Step 1: Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_features = 28 * 28
key_feature_weights = torch.ones(num_features)

# Define key pixels for each class (important features)
key_features = {
    0: [14 * 28 + 14],  # Center for 0
    1: [13 * 28 + 14],  # Vertical middle-left for 1
    2: [7 * 28 + 6],    # Top-left curve for 2
    3: [7 * 28 + 13],   # Top curve for 3
    4: [14 * 28 + 7],   # Middle horizontal line for 4
    5: [7 * 28 + 7, 21 * 28 + 7],  # Left and bottom curves for 5
    6: [21 * 28 + 7],   # Bottom loop for 6
    7: [7 * 28 + 14],   # Top horizontal for 7
    8: [7 * 28 + 7, 21 * 28 + 7],  # Upper and lower loops for 8
    9: [21 * 28 + 7],   # Bottom loop for 9
}

# Assign high weights to key pixels for each class
for i in range(10):
    for pixel in key_features[i]:
        key_feature_weights[pixel] = 10  # High weight for key pixels

# Convert to tensor
key_feature_weights = torch.tensor(key_feature_weights, dtype=torch.float32)
# Step 2: Flatten the images and apply TF-IDF weights
def apply_tfidf_weights(images, key_feature_weights):
    # Flatten the images to 1D and apply the TF-IDF weights
    images_flat = images.view(images.size(0), -1)  # Flatten to [batch_size, 784]
    return images_flat * key_feature_weights

# Convert train and test data into tensors
train_images = torch.stack([image[0] for image in train_dataset])
train_labels = torch.tensor([image[1] for image in train_dataset])

test_images = torch.stack([image[0] for image in test_dataset])
test_labels = torch.tensor([image[1] for image in test_dataset])

# Step 3: Apply the TF-IDF weights (assuming key_feature_weights is already computed)
key_feature_weights = torch.tensor(key_feature_weights, dtype=torch.float32)  # Assuming this is precomputed

train_images_tfidf = apply_tfidf_weights(train_images, key_feature_weights)
test_images_tfidf = apply_tfidf_weights(test_images, key_feature_weights)

# Step 4: Flatten the images for the classifier
train_images_tfidf = train_images_tfidf.view(train_images_tfidf.size(0), -1).numpy()  # Convert to NumPy array
test_images_tfidf = test_images_tfidf.view(test_images_tfidf.size(0), -1).numpy()

# Step 5: Train a classifier (SVM example)
clf = SVC(kernel='linear')  # You can try other classifiers like logistic regression
clf.fit(train_images_tfidf, train_labels.numpy())

# Step 6: Evaluate the classifier
test_predictions = clf.predict(test_images_tfidf)
accuracy = accuracy_score(test_labels.numpy(), test_predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
