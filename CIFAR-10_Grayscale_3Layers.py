  # Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

def rgb_to_grayscale(images):
    return np.dot(images[...,:3], [0.2989, 0.5870, 0.1140])
# Normalize pixel values between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = rgb_to_grayscale(train_images)
test_images = rgb_to_grayscale(test_images)
print(train_images.shape)
# \/

# Class names in English
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
# Build the CNN model
model = models.Sequential()#54,
model.add(layers.Conv2D(90, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(180, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(360, (3, 3), activation='relu'))

# Add fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(240 , activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Using softmax for multi-class classification

# Print the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # Softmax outputs probabilities, so from_logits=False
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Plot training results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Example: Display the first image in the test set and predict its label
# import numpy as np
#
# plt.figure(figsize=(2, 2))
# plt.imshow(test_images[0])
# plt.title("Actual: " + class_names[test_labels[0][0]])

# Predict the label
predictions = model.predict(test_images)
predicted_label = np.argmax(predictions[0])

# Print the predicted label
print(f"Predicted label: {class_names[predicted_label]}")


plt.figure(figsize=(2, 2))
plt.imshow(test_images[0])
plt.show()
plt.title("Actual: " + class_names[test_labels[0][0]])