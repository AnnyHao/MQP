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


# Normalize image data (assuming pixel values between 0-255)
def normalize(X):
    return X / 255.0


# Train and test split
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Softmax regression using LogisticRegression from sklearn
def softmax_regression(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=1000)
    model.fit(X_train, y_train)
    return model


# Measure accuracy
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Save softmax probabilities with renormalization
def save_softmax_probabilities_with_renormalization(model, X_test, output_csv):
    # Get softmax probabilities
    softmax_probs = model.predict_proba(X_test)

    # Create a DataFrame for the probabilities
    df = pd.DataFrame(softmax_probs, columns=[f'Class_{i}' for i in range(softmax_probs.shape[1])])

    # Calculate the sum of each row
    df['Row_Sum'] = df.sum(axis=1)

    # Renormalize probabilities so each row sums to 1
    df.iloc[:, :-1] = df.iloc[:, :-1].div(df['Row_Sum'], axis=0)

    # Recompute the row sum after renormalization (should be exactly 1 now)
    df['Row_Sum'] = df.sum(axis=1)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Softmax probabilities and renormalized row sums saved to {output_csv}")


# Main function to run the program on .npz files
def main_npz(img_file, label_file, output_csv):
    X, y = load_npz_data(img_file, label_file)  # Load data from .npz files
    X = normalize(X)  # Normalize image data

    X_train, X_test, y_train, y_test = split_data(X, y)  # Split into train and test sets

    model = softmax_regression(X_train, y_train)  # Train softmax regression model
    accuracy = evaluate_model(model, X_test, y_test)  # Evaluate accuracy
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Save softmax probabilities and renormalized row sums to CSV
    save_softmax_probabilities_with_renormalization(model, X_test, output_csv)


# Example usage for npz files
img_file = 'k49-train-imgs.npz'
label_file = 'k49-train-labels.npz'
output_csv = 'softmax_output_with_renormalization_npz2.csv'
main_npz(img_file, label_file, output_csv)