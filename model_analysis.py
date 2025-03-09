import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from model import create_model, train_model
from utils import prepare_for_prediction, preprocess_image

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def create_improved_model():
    """Create an improved model with techniques to reduce overfitting"""
    model = Sequential(
        [
            # First convolutional block
            Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1),
            ),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Second convolutional block
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Fully connected layers
            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )

    # Use a better optimizer with learning rate scheduling
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def train_improved_model():
    """Train an improved model with data augmentation and other techniques"""
    print("\nTraining improved model...")

    # Load MNIST dataset
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

    # Create data generator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode="nearest",
    )

    # Create improved model
    model = create_improved_model()

    # Callbacks for better training
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001, verbose=1
    )

    # Train model with data augmentation
    history = model.fit(
        datagen.flow(x_train, y_train_cat, batch_size=128),
        epochs=30,
        validation_data=(x_test, y_test_cat),
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    # Save improved model
    model.save("improved_digit_model.keras")
    print("Improved model saved to improved_digit_model.keras")

    # Plot training history
    plot_training_history(history)

    # Evaluate improved model
    test_loss, test_acc = model.evaluate(x_test, y_test_cat)
    print(f"Improved model test accuracy: {test_acc:.4f}")

    # Make predictions with improved model
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Create confusion matrix for improved model
    conf_matrix = confusion_matrix(y_test, y_pred_classes)

    # Plot confusion matrix for improved model
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(10),
        yticklabels=range(10),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Improved Model Confusion Matrix")
    plt.savefig("improved_confusion_matrix.png")
    plt.close()

    return model, history


def plot_training_history(history):
    """Plot training history with loss and accuracy curves"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()


if __name__ == "__main__":
    improved_model, history = train_improved_model()
    
    print("\nTraining complete! The improved model has been saved.")