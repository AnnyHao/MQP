import numpy as np
import pandas as pd
from scipy.special import softmax


def load_data(dataset_type):
    if dataset_type == 'mnist':
        train_data = pd.read_csv('mnist_train.csv')
        test_data = pd.read_csv('mnist_test.csv')

        X_test = test_data.drop(columns='label').values
        y_test = test_data['label'].values

        X_test = X_test / 255.0

        return X_test, y_test

    elif dataset_type == 'k49':
        train_imgs = np.load('k49-train-imgs.npz')['arr_0']
        train_labels = np.load('k49-train-labels.npz')['arr_0']
        test_imgs = np.load('k49-test-imgs.npz')['arr_0']
        test_labels = np.load('k49-test-labels.npz')['arr_0']

        X_test = test_imgs
        y_test = test_labels

        return X_test, y_test

    else:
        raise ValueError("Invalid dataset type. Choose 'mnist' or 'k49'.")


# Specify the dataset
dataset_type = 'mnist'  # 'mnist' or 'k49'

for x in range(50):
    X_test, y_test = load_data(dataset_type)

    num_samples = X_test.shape[0]
    num_classes = 10  # 49 if k49, 10 if MNIST
    logits = np.random.rand(num_samples, num_classes)

    probabilities = softmax(logits, axis=1)

    probabilities_df = pd.DataFrame(probabilities, columns=[f'Class_{i}' for i in range(num_classes)])
    probabilities_df['True_Label'] = y_test
    probabilities_df.to_csv(f'{dataset_type}_probabilities.csv', index=False)

    predicted_classes = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predicted_classes == y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')
