import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load data from .npz files
def load_npz_data(img_file, label_file):
    # Load image and label data from npz files
    with np.load(img_file) as img_data, np.load(label_file) as label_data:
        X = img_data['arr_0']  # Assuming the key for images is 'arr_0'
        y = label_data['arr_0']  # Assuming the key for labels is 'arr_0'

    # Flatten the image data if it's 3D (i.e., num_samples x height x width)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)  # Flatten each image to 1D (num_samples, height * width)

    return X, y


def normalize(X):
    return X / 255.0


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Softmax regression using LogisticRegression from sklearn
def softmax_regression(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Save softmax probabilities with renormalization
def save_softmax_probabilities_with_renormalization(model, X_test, output_csv):
    softmax_probs = model.predict_proba(X_test)

    df = pd.DataFrame(softmax_probs, columns=[f'Class_{i}' for i in range(softmax_probs.shape[1])])

    df['Row_Sum'] = df.sum(axis=1)

    df.iloc[:, :-1] = df.iloc[:, :-1].div(df['Row_Sum'], axis=0)

    df['Row_Sum'] = df.sum(axis=1)

    df.to_csv(output_csv, index=False)
    print(f"Softmax probabilities and renormalized row sums saved to {output_csv}")


def main_npz(img_file, label_file, output_csv):
    X, y = load_npz_data(img_file, label_file)
    X = normalize(X)  # Normalize image data

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = softmax_regression(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # save_softmax_probabilities_with_renormalization(model, X_test, output_csv)


# Example usage for npz files
img_file = 'k49-train-imgs.npz'
label_file = 'k49-train-labels.npz'
output_csv = 'softmax_output_with_renormalization_npz3.csv'
main_npz(img_file, label_file, output_csv)
