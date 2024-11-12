
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


## License
This project is licensed under the MIT License. See the `LICENSE` file for more details

## Acknowledgements
This project was funded by the German University of Technology in Muscat, Sultanate of Oman.
