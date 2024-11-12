
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
import pydicom
import os

def load_dicom_images(directory_path):
    """
    Load DICOM images from a specified directory.

    Parameters:
        directory_path (str): Path to the directory containing DICOM files.

    Returns:
        images (list): List of loaded images as NumPy arrays.
    """
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".dcm"):
            filepath = os.path.join(directory_path, filename)
            dicom_file = pydicom.dcmread(filepath)
            image = dicom_file.pixel_array
            images.append(image)
    return images

def preprocess_image(image):
    """
    Preprocess an image by resizing, normalizing, and reducing noise.

    Parameters:
        image (numpy.ndarray): The input image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Resize the image to 256x256 pixels
    image = cv2.resize(image, (256, 256))

    # Normalize the image to the range [0, 1]
    image = image / 255.0

    # Apply Gaussian filter for noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)

    return image

def augment_images(images):
    """
    Augment a list of images using rotation, flipping, and zooming.

    Parameters:
        images (list): List of images to augment as NumPy arrays.

    Returns:
        augmented_images (list): List of augmented images as NumPy arrays.
    """
    data_augmentation = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    augmented_images = []
    for image in images:
        image = np.expand_dims(image, axis=0)  # Expand dimensions for augmentation
        augmented_iter = data_augmentation.flow(image, batch_size=1)
        for i in range(5):  # Generate 5 augmented versions per image
            augmented_image = augmented_iter.next()[0]
            augmented_images.append(augmented_image)

    return augmented_images

def save_preprocessed_images(images, output_directory):
    """
    Save preprocessed images to a specified output directory.

    Parameters:
        images (list): List of images to save as NumPy arrays.
        output_directory (str): Path to the directory where images will be saved.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, image in enumerate(images):
        output_path = os.path.join(output_directory, f"preprocessed_image_{i}.png")
        cv2.imwrite(output_path, (image * 255).astype(np.uint8))  # Save image in [0, 255] range

def main():
    # Define directories for input and output
    input_directory = "./dicom_images"
    output_directory = "./preprocessed_images"

    # Load DICOM images
    images = load_dicom_images(input_directory)

    # Preprocess images
    preprocessed_images = [preprocess_image(image) for image in images]

    # Augment images
    augmented_images = augment_images(preprocessed_images)

    # Save preprocessed and augmented images
    save_preprocessed_images(augmented_images, output_directory)

if __name__ == "__main__":
    main()
