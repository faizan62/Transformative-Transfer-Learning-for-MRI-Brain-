
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_architecture import build_resnet50_model

def load_data(data_directory, target_size=(256, 256), batch_size=32):
    """
    Load training and validation data using ImageDataGenerator.

    Parameters:
        data_directory (str): Path to the directory containing training and validation subdirectories.
        target_size (tuple): Size to resize images to.
        batch_size (int): Batch size for data loading.

    Returns:
        train_generator, validation_generator: Data generators for training and validation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        data_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

def train_model(data_directory, model_save_path, target_size=(256, 256), batch_size=32, epochs=50, learning_rate=0.001):
    """
    Train the model using training and validation data.

    Parameters:
        data_directory (str): Path to the directory containing training and validation data.
        model_save_path (str): Path to save the trained model.
        target_size (tuple): Size to resize images to.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Load data
    train_generator, validation_generator = load_data(data_directory, target_size, batch_size)

    # Build model
    model = build_resnet50_model(input_shape=(target_size[0], target_size[1], 3), learning_rate=learning_rate)

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True)

    # Train model
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

def main():
    # Define directories and parameters
    data_directory = "./data"
    model_save_path = "./saved_models/brain_tumor_classifier.h5"
    target_size = (256, 256)
    batch_size = 32
    epochs = 50
    learning_rate = 0.001

    # Train the model
    train_model(data_directory, model_save_path, target_size, batch_size, epochs, learning_rate)

if __name__ == "__main__":
    main()
