import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler


# Load CIFAR-10 data
def load_cifar10_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()  # Flatten labels for compatibility with scikit-learn
    y_test = y_test.flatten()
    return X_train, X_test, y_train, y_test


# Normalize pixel values
def normalize(X):
    return X.astype('float32') / 255.0


# Flatten images (32x32x3) to (3072,) for each image
def flatten(X):
    return X.reshape(X.shape[0], -1)


# Train the softmax regression model
def softmax_regression(X_train, y_train):
    model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=3000)
    model.fit(X_train, y_train)
    return model


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Save softmax probabilities
def save_softmax_probabilities(model, X_test, output_csv):
    softmax_probs = model.predict_proba(X_test)
    df = pd.DataFrame(softmax_probs, columns=[f'Class_{i}' for i in range(softmax_probs.shape[1])])
    df['Row_Sum'] = df.sum(axis=1)
    df.to_csv(output_csv, index=False)
    print(f"Softmax probabilities saved to {output_csv}")


# Main function
def main(output_csv):
    accuracies = []

    # Loop to run the process 4 times
    for i in range(4):
        X_train, X_test, y_train, y_test = load_cifar10_data()
        X_train = normalize(flatten(X_train))
        X_test = normalize(flatten(X_test))

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = softmax_regression(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)

        accuracies.append(accuracy)
        print(f"Run {i + 1}: Accuracy = {accuracy * 100:.2f}%")

    # Calculate and print the average accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"Average Accuracy after 4 runs: {avg_accuracy * 100:.2f}%")

    # Save softmax probabilities for the last run
    # save_softmax_probabilities(model, X_test, output_csv)


# Output CSV file for softmax probabilities
output_csv = r'cifar10_output.csv'
main(output_csv)
