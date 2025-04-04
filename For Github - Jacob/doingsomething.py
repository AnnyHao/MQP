import numpy as np
import pandas as pd
from scipy.special import softmax


def load_data(dataset_type):
    if dataset_type == 'mnist':
        # Load the MNIST dataset
        test_data = pd.read_csv('mnist_test.csv')

        # Separate features and labels
        X_test = test_data.drop(columns='label').values
        y_test = test_data['label'].values

        # Normalize the data
        X_test = X_test / 255.0

        return X_test, y_test

    elif dataset_type == 'k49':
        # Load the K49 dataset
        test_imgs = np.load('k49-test-imgs.npz')['arr_0']
        test_labels = np.load('k49-test-labels.npz')['arr_0']

        return test_imgs, test_labels

    else:
        raise ValueError("Invalid dataset type. Choose 'mnist' or 'k49'.")


# Specify the dataset you want to use ('mnist' or 'k49')
dataset_type = 'mnist'

# Load the dataset
X_test, y_test = load_data(dataset_type)

# Flatten each image (to treat all pixel values as logits)
num_samples = X_test.shape[0]
X_test_flat = X_test.reshape(num_samples, -1)  # Flatten each image into a 1D array

# Apply softmax to the flattened pixel values
probabilities = softmax(X_test_flat, axis=1)  # Apply softmax along the image's pixel values

# Save probabilities to CSV
probabilities_df = pd.DataFrame(probabilities)
probabilities_df['True_Label'] = y_test
probabilities_df.to_csv(f'{dataset_type}_pixel_softmax.csv', index=False)

# Get "predicted" label as the index of the maximum probability (pixel index)
predicted_labels = np.argmax(probabilities, axis=1)

# Calculate accuracy by comparing predicted "labels" with true labels
accuracy = np.mean(predicted_labels == y_test)

# Print accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
