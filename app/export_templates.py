

class export_code:
    def __init__(self):
        self.architecture_choice = None
        self.complexity_choice = None

    def export_basic_vgg16_script(self, responses, best_hyperparameters, file_name="custom_cnn_model.py"):
        """
        Generates and exports a basic vgg16 Python script for a CNN model.

        Parameters:
        - responses: Dictionary containing values for 'num_classes', 'img_size', and 'channels'.
        - best_hyperparameters: Dictionary containing the best hyperparameters.
        - file_name: The name of the Python script to create.
        """
        # Extracting values from responses
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
        data_setpath = responses[2]
        input_shape = (img_size, img_size, channels)  # (height, width, channels)
        # Hyperparameters
        filters = best_hyperparameters['filters']
        dropout_rate = best_hyperparameters['dropout_rate']
        learning_rate = best_hyperparameters['learning_rate']

        # Model script template
        script_content = f"""\
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming responses are passed correctly
dataset_path = '{data_setpath}'
num_classes = {num_classes}
img_size = {img_size}
channels = {channels}
filters = {filters}
dropout_rate = {dropout_rate}
learning_rate = {learning_rate}

class basicModel:
    def load_dataset(self, dataset_path, img_size):
        # Example function for loading a dataset
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            dataset_path + '/train',
            target_size=(img_size, img_size),
            batch_size=32,  
            class_mode='categorical')

        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = validation_datagen.flow_from_directory(
            dataset_path + '/val',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='categorical')

        return train_generator, validation_generator

    def build_VGG16_model(self,num_classes, img_size, channels , filters=64, dropout_rate=0.5, learning_rate=1e-3):

        # responses = [int(image_size), int(num_classes), dataset_path, int(rgb_or_grey), suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Assuming responses[0] contains the image size and your images are RGB
        input_shape = (img_size, img_size, channels)  # (height, width, channels)

        model = tf.keras.models.Sequential([
            # Input layer
            tf.keras.layers.InputLayer(input_shape=input_shape),

            # Block 1
            tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Optional: Dropout layer after max pooling
            tf.keras.layers.Dropout(dropout_rate),

            # Block 2
            tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(dropout_rate),

            # Block 3
            tf.keras.layers.Conv2D(filters * 4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(filters * 4, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(dropout_rate),

            # Flattening and Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])

        # Compile the model with the dynamic learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


    if __name__ == "__main__":
        train_generator, validation_generator = load_dataset(dataset_path, img_size)
        model = build_VGG16_model(num_classes, img_size, channels, filters, dropout_rate, learning_rate)
        model.fit(train_generator, validation_data=validation_generator, epochs=10)
        model.save('trained_model.h5')
        print("Model training complete and saved.")
    """

        print("Image Classification CNN Skeleton:\n", script_content)

        # Write the script to a file
        with open(file_name, "w") as file:
            file.write(script_content)

        print(f"Custom CNN model script has been saved to {file_name}.")

    def export_basic_resNet_script(self, responses, best_hyperparameters, file_name="custom_cnn_model.py"):
        """
        Generates and exports a Basic resNet Python script for a CNN model to get started.

        Parameters:
        - responses: Dictionary containing values for 'num_classes', 'img_size', and 'channels'.
        - best_hyperparameters: Dictionary containing the best hyperparameters.
        - file_name: The name of the Python script to create.
        """
        # Extracting values from responses
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
        data_setpath = responses[2]
        input_shape = (img_size, img_size, channels)  # (height, width, channels)
        # Hyperparameters
        filters = best_hyperparameters['filters']
        dropout_rate = best_hyperparameters['dropout_rate']
        learning_rate = best_hyperparameters['learning_rate']

        # Model script template
        script_content = f"""\
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming responses are passed correctly
dataset_path = '{data_setpath}'
num_classes = {num_classes}
img_size = {img_size}
channels = {channels}
filters = {filters}
dropout_rate = {dropout_rate}
learning_rate = {learning_rate}

class basicModel:
    def load_dataset(self, dataset_path, img_size):
        # Example function for loading a dataset
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            dataset_path + '/train',
            target_size=(img_size, img_size),
            batch_size=32,  
            class_mode='categorical')

        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = validation_datagen.flow_from_directory(
            dataset_path + '/val',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='categorical')

        return train_generator, validation_generator
    def residual_block(self, x, filters, conv_num=3, stride=1):
        shortcut = x
        for i in range(conv_num):
            x = layers.Conv2D(filters, (3, 3), padding='same', strides=stride if i == 0 else 1)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        if stride != 1 or filters != shortcut.shape[-1]:
            shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def build_ResNet_model(self,num_classes, img_size, channels, learning_rate=1e-3):
        input_shape = (img_size, img_size, channels)

        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(64, (7, 7), strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

        # Add residual blocks
        x = self.residual_block(x, 64, conv_num=3, stride=1)
        x = self.residual_block(x, 128, conv_num=4, stride=2)
        x = self.residual_block(x, 256, conv_num=6, stride=2)
        x = self.residual_block(x, 512, conv_num=3, stride=2)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    if __name__ == "__main__":
        train_generator, validation_generator = load_dataset(dataset_path, img_size)
        model = build_ResNet_model(num_classes, img_size, channels, learning_rate)
        model.fit(train_generator, validation_data=validation_generator, epochs=10)
        model.save('trained_model.h5')
        print("Model training complete and saved.")
    """

        print("Image Classification CNN Skeleton:\n", script_content)

        # Write the script to a file
        with open(file_name, "w") as file:
            file.write(script_content)

        print(f"Custom CNN model script has been saved to {file_name}.")

    def export_basic_leNet_Script(self, responses, best_hyperparameters, file_name="custom_cnn_model.py"):
        """
        Generates and exports a basic leNet Python script for a CNN model.

        Parameters:
        - responses: Dictionary containing values for 'num_classes', 'img_size', and 'channels'.
        - best_hyperparameters: Dictionary containing the best hyperparameters.
        - file_name: The name of the Python script to create.
        """
        # Extracting values from responses
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
        data_setpath = responses[2]
        input_shape = (img_size, img_size, channels)  # (height, width, channels)
        # Hyperparameters
        filters = best_hyperparameters['filters']
        dropout_rate = best_hyperparameters['dropout_rate']
        learning_rate = best_hyperparameters['learning_rate']

        # Model script template
        script_content = f"""\
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming responses are passed correctly
dataset_path = '{data_setpath}'
num_classes = {num_classes}
img_size = {img_size}
channels = {channels}
filters = {filters}
dropout_rate = {dropout_rate}
learning_rate = {learning_rate}


class basicModel:
    def load_dataset(self, dataset_path, img_size):
        # Example function for loading a dataset
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            dataset_path + '/train',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='categorical')

        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = validation_datagen.flow_from_directory(
            dataset_path + '/val',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='categorical')

        return train_generator, validation_generator

    def build_leNet_model(self, num_classes, img_size, channels, learning_rate=1e-3):
        # responses = [int(image_size), int(num_classes), dataset_path, int(rgb_or_grey), suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Assuming responses[0] contains the image size and your images are RGB
        input_shape = (img_size, img_size, channels)  # (height, width, channels)

        model = tf.keras.models.Sequential([
            # C1: Convolutional layer with 6 filters, each of size 5x5.
            tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu',
                                   input_shape=input_shape, padding="same"),
            # S2: Subsampling/Pooling layer.
            tf.keras.layers.AveragePooling2D(),

            # C3: Convolutional layer with 16 filters, each of size 5x5.
            tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'),
            # S4: Subsampling/Pooling layer.
            tf.keras.layers.AveragePooling2D(),

            # C5: Fully connected convolutional layer with 120 filters.
            tf.keras.layers.Conv2D(120, kernel_size=(5, 5), activation='relu'),

            # Flatten the convolutions to feed them into fully connected layers
            tf.keras.layers.Flatten(),

            # F6: Fully connected layer with 84 units.
            tf.keras.layers.Dense(84, activation='relu'),

            # Output layer with a unit for each class.
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model with the dynamic learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    if __name__ == "__main__":
        train_generator, validation_generator = load_dataset(dataset_path, img_size)
        model = build_leNet_model(num_classes, img_size, channels, learning_rate)
        model.fit(train_generator, validation_data=validation_generator, epochs=10)
        model.save('trained_model.h5')
        print("Model training complete and saved.")
        """

        print("Image Classification CNN Skeleton:\n", script_content)

        # Write the script to a file
        with open(file_name, "w") as file:
            file.write(script_content)

        print(f"Custom CNN model script has been saved to {file_name}.")

    def export_basic_alexNet_Script(self, responses, best_hyperparameters, file_name="custom_cnn_model.py"):
        """
        Generates and exports a basic AlexNet Python script for a CNN model.

        Parameters:
        - responses: Dictionary containing values for 'num_classes', 'img_size', and 'channels'.
        - best_hyperparameters: Dictionary containing the best hyperparameters.
        - file_name: The name of the Python script to create.
        """
        # Extracting values from responses
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
        data_setpath = responses[2]
        input_shape = (img_size, img_size, channels)  # (height, width, channels)
        # Hyperparameters
        filters = best_hyperparameters['filters']
        dropout_rate = best_hyperparameters['dropout_rate']
        learning_rate = best_hyperparameters['learning_rate']

        # Model script template
        script_content = f"""\
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Assuming responses are passed correctly
dataset_path = '{data_setpath}'
num_classes = {num_classes}
img_size = {img_size}
channels = {channels}
filters = {filters}
dropout_rate = {dropout_rate}
learning_rate = {learning_rate}


class basicModel:
    def load_dataset(self, dataset_path, img_size):
        # Example function for loading a dataset
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        train_generator = train_datagen.flow_from_directory(
            dataset_path + '/train',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='categorical')

        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_generator = validation_datagen.flow_from_directory(
            dataset_path + '/val',
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='categorical')

        return train_generator, validation_generator

    def build_alexNet_model(self, num_classes, img_size, channels, learning_rate=1e-3):
        # responses = [int(image_size), int(num_classes), dataset_path, int(rgb_or_grey), suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Assuming responses[0] contains the image size and your images are RGB
        input_shape = (img_size, img_size, channels)  # (height, width, channels)

        model = tf.keras.models.Sequential([
            # First Convolutional Layer
            tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(2, 2), activation='relu',
                                   input_shape=(img_size, img_size, channels), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

            tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

            # Reducing the number of layers to adapt to smaller image sizes
            tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),

            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),

            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])

        # Compile the model with the dynamic learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    if __name__ == "__main__":
        train_generator, validation_generator = load_dataset(dataset_path, img_size)
        model = build_alexNet_model(num_classes, img_size, channels, learning_rate)
        model.fit(train_generator, validation_data=validation_generator, epochs=10)
        model.save('trained_model.h5')
        print("Model training complete and saved.")
        """

        print("Image Classification CNN Skeleton:\n", script_content)

        # Write the script to a file
        with open(file_name, "w") as file:
            file.write(script_content)

        print(f"Custom CNN model script has been saved to {file_name}.")

