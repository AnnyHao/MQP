import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load data from CSV
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    X = data.drop('label', axis=1).values  # Features (pixel data)
    y = data['label'].values  # Labels (digit or class)
    return X, y


# Normalize pixel values (assuming values are in the range 0-255)
def normalize(X):
    return X / 255.0


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def softmax_regression(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def save_softmax_probabilities(model, X_test, output_csv):
    softmax_probs = model.predict_proba(X_test)  # Get softmax probabilities
    df = pd.DataFrame(softmax_probs, columns=[f'Class_{i}' for i in range(softmax_probs.shape[1])])
    df['Row_Sum'] = df.sum(axis=1)
    df.to_csv(output_csv, index=False)  # Save to CSV
    print(f"Softmax probabilities saved to {output_csv}")


def main(csv_path, output_csv):
    X, y = load_data(csv_path)
    X = normalize(X)  # Normalize pixel data
    X_train, X_test, y_train, y_test = split_data(X, y)  # Split into train and test sets

    model = softmax_regression(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    save_softmax_probabilities(model, X_test, output_csv)


csv_path = r"C:\Users\jacob\Desktop\mnist_train.csv"
output_csv = r"C:\Users\jacob\Desktop\output.csv"
main(csv_path, output_csv)
