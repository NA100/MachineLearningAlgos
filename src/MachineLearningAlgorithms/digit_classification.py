import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")

"""
Sure! Here's a simple neural network example using TensorFlow and Keras in Python.
We'll build a small network to classify digits from the MNIST dataset (handwritten digits 0–9).
This is a classic beginner-friendly deep learning project.

🔍 What’s Happening Here?
Input: 28×28 pixel images (flattened into 784 values)
Hidden Layers:
First: 128 neurons, ReLU activation
Second: 64 neurons, ReLU
Output Layer: 10 neurons (for digits 0–9), softmax activation to give probabilities
Loss Function: Categorical crossentropy (since it’s multi-class classification)
Optimizer: Adam (a fast, popular choice)
Epochs: Runs training 5 times over the dataset
"""