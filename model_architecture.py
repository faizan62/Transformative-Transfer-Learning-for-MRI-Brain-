
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_vgg16_model(input_shape=(256, 256, 3), learning_rate=0.001):
    """
    Build and compile a VGG16-based model for MRI brain tumor classification.

    Parameters:
        input_shape (tuple): Shape of the input image.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled VGG16 model.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_resnet50_model(input_shape=(256, 256, 3), learning_rate=0.001):
    """
    Build and compile a ResNet50-based model for MRI brain tumor classification.

    Parameters:
        input_shape (tuple): Shape of the input image.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled ResNet50 model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_inceptionv3_model(input_shape=(256, 256, 3), learning_rate=0.001):
    """
    Build and compile an InceptionV3-based model for MRI brain tumor classification.

    Parameters:
        input_shape (tuple): Shape of the input image.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled InceptionV3 model.
    """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_densenet121_model(input_shape=(256, 256, 3), learning_rate=0.001):
    """
    Build and compile a DenseNet121-based model for MRI brain tumor classification.

    Parameters:
        input_shape (tuple): Shape of the input image.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        Model: Compiled DenseNet121 model.
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Example: Build and compile the ResNet50 model
    model = build_resnet50_model()
    model.summary()

if __name__ == "__main__":
    main()
