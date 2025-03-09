import os

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


def create_model():
    """
    Create a CNN model for digit recognition

    Returns:
        Compiled Keras model
    """
    model = Sequential(
        [
            # First convolutional layer
            Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            # Second convolutional layer
            Conv2D(64, kernel_size=(3, 3), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            # Flatten data
            Flatten(),
            # Fully connected layer
            Dense(128, activation="relu"),
            Dropout(0.5),  # Dropout to reduce overfitting
            # Output layer with 10 neurons (0-9)
            Dense(10, activation="softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_model(
    model=None, epochs=10, batch_size=128, save_path="improved_digit_model.keras"
):
    """
    Train the model on MNIST dataset

    Args:
        model: Model to train (if None, a new model will be created)
        epochs: Number of training epochs
        batch_size: Batch size
        save_path: Path to save the model

    Returns:
        Trained model and training history
    """
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # One-hot encoding for labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Create model if not provided
    if model is None:
        model = create_model()

    # Train model
    history = model.fit(
        x_train,
        y_train_cat,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_cat),
        verbose=1,
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test_cat)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model with new Keras format
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model, history


def load_trained_model(model_path="improved_digit_model.keras"):
    """
    Load trained model

    Args:
        model_path: Path to model file

    Returns:
        Loaded model
    """
    # Check both .keras and .h5 files for backward compatibility
    if os.path.exists(model_path):
        return load_model(model_path)
    elif os.path.exists(model_path.replace(".keras", ".h5")):
        # Compatible with old file
        old_path = model_path.replace(".keras", ".h5")
        print(f"Loading legacy model from {old_path}")
        model = load_model(old_path)
        # Save with new format
        model.save(model_path)
        return model
    else:
        print(f"Model not found at {model_path}. Training a new model...")
        model = create_model()
        model, _ = train_model(model, save_path=model_path)
        return model


def predict_digit(model, image):
    """
    Predict digit from image

    Args:
        model: Trained model
        image: Input image (processed)

    Returns:
        Predicted digit and probability
    """
    # Predict
    predictions = model.predict(image)

    # Get digit with highest probability
    digit = np.argmax(predictions[0])
    probability = predictions[0][digit]

    return digit, probability
