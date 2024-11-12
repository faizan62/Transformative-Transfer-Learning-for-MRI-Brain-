
# Transformative Transfer Learning for MRI Brain Tumor Classification

This project implements the Transformative Transfer Learning (TTL) model for MRI brain tumor classification, utilizing pre-trained deep learning models such as VGG16, ResNet50, InceptionV3, and DenseNet-121. The project aims to improve the accuracy of brain tumor classification and assist in clinical diagnostics.

## Project Overview
- **Data Preprocessing**: Resize images, normalize, noise reduction using Gaussian filter, and augment data.
- **Model Architectures**: Utilizes several pre-trained models (VGG16, ResNet50, InceptionV3, DenseNet-121).
- **Training**: Train the model using data generators and validate performance using metrics like accuracy, precision, recall, and F1-score.

## Directory Structure
```
TransformativeTransferLearning_MRI/
├── data_preprocessing.py
├── model_architecture.py
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
├── LICENSE
└── saved_models/
    └── transformative_transfer_learning_model.h5
```

## Requirements
Install the required packages using:
```
pip install -r requirements.txt
```

## Usage
1. **Data Preprocessing**:
   - Use `data_preprocessing.py` to preprocess and augment the dataset.
2. **Model Training**:
   - Use `train.py` to train the model on the MRI dataset.
3. **Evaluation**:
   - Use `evaluate.py` (to be implemented) to evaluate the model.

## Dataset
The dataset used in this study includes MRI scans from the Nickparvar and Cheng datasets. Make sure to download and place the dataset in the `./data` directory.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Authors
- Raja Waseem Anwar, Mohammad Abrar, Faizan Ullah, Sabir Shah

## Acknowledgements
This project was funded by the German University of Technology in Muscat, Sultanate of Oman.
