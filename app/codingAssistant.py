import optuna

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0  # You can choose from B0 to B7
from tensorflow.keras.models import Model
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models

from flask import Flask, render_template, request

class CodingAssistant:

    def __init__(self):
        self.architecture_choice = None
        self.complexity_choice = None
        self.best_accuracy = 0

    def start_interaction(self):
        print("Please select an option:")
        start = self.get_valid_input("Would you like to get started? (y/n): ",
                                                 lambda x: x.lower() in ['y', 'n'])
        if start == 'y':
            self.customise_image_classification_param(image_size=32, num_classes=3,rgb_or_grey=3,dataset_path='C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)',hyperparam_tuning='y',hyp_choice=3,arch_choice=2,n_trials=20 );


    def get_valid_input(self, prompt, validation_func, error_message="Invalid answer. Please try again."):
        while True:
            user_input = input(prompt)
            if validation_func(user_input):
                return user_input
            print(error_message)

    # def customise_image_classification_param(self):
    #     basic_temp= 'basic template.py'
    #     image_size = self.get_valid_input("Enter image size (e.g., 128 for 128x128 pixels): ",
    #                                           lambda x: x.isdigit() and int(x) > 0)
    #
    #     num_classes = self.get_valid_input("How many classes would you like? (e.g., 10): ",
    #                                        lambda x: x.isdigit() and int(x) > 0)
    #
    #     rgb_or_grey = self.get_valid_input("Do you want to use rgb or greyscale? (rgb = 3 / greyscale = 1) ",
    #                                        lambda x: x.isdigit() and int(x) in [1, 3])
    #
    #     # Prompt user for the path to their dataset
    #     dataset_path = self.get_valid_input(
    #         "Enter the path to your dataset or leave blank if not applicable): ",
    #         lambda x: True)  # Validation allows any input, including blank for flexibility
    #
    #     responses = [int(image_size), int(num_classes), dataset_path, int(rgb_or_grey),0]
    #
    #     # responses = [32, 3, 'C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)',3]
    #     # C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)
    #     hyperparam_tuning = self.get_valid_input("Would you like your hyperparameters to be tuned? (y/n): ",
    #                                              lambda x: x.lower() in ['y', 'n'])
    #
    #     # hyperparam_tuning = 'y'
    #     if hyperparam_tuning == 'y':
    #         hyp_choice = self.get_valid_input("Select a hyperparameter optimization method:\n1: Optuna\n2: Grid Search"
    #                                           "\n3: Random Search\n4: Bayesian Optimization\nEnter your choice (1-4): ",
    #                                           lambda x: x.isdigit() and int(x) in [1, 2, 3, 4])
    #         # responses.append(int(hyp_choice))
    #         responses[4] = int(hyp_choice)
    #
    #
    #     arch_choice = self.get_valid_input("Select a CNN architecture:\n1: ResNet\n2: VGG16"
    #                                        "\n3: leNet\n4: alexNet\nEnter your choice (1-4): ",
    #                                        lambda x: x.isdigit() and int(x) in [1, 2, 3, 4])
    #     responses.append(int(arch_choice))
    #
    #     best_params = self.choose_hyperparameter_optimization_method(responses)
    #
    #     if arch_choice == '1':
    #         export_code.export_basic_resNet_script(responses, best_params, basic_temp)
    #     if arch_choice == '2':
    #         export_code.export_basic_vgg16_script(responses, best_params, basic_temp)
    #     if arch_choice == '3':
    #         export_code.export_basic_leNet_Script(responses, best_params, basic_temp)
    #     if arch_choice == '4':
    #         export_code.export_basic_alexNet_Script(responses, best_params, basic_temp)

    def customise_image_classification_param(self, image_size, num_classes, rgb_or_grey, dataset_path, hyperparam_tuning, hyp_choice, arch_choice, n_trials):
        basic_temp= 'basic template.py'
        try:
            responses = [int(image_size), int(num_classes), dataset_path, int(rgb_or_grey),int(hyp_choice), int(arch_choice), int(n_trials)]
            # responses = [32, 3, 'C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)',3,1,2]
            print("_________________ inside customise_image_classification_param ____________________ ")
            # responses = [32, 3, 'C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)',3]
            # C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)


            # responses.append(int(hyp_choice))
            if hyperparam_tuning.lower() == 'y':
                best_params = self.choose_hyperparameter_optimization_method(responses, n_trials)
                print("_________________ supposed to return the best param here ____________________")
                print(best_params)
                # basic_temp = 'basic template.py'
                # if arch_choice == '1':
                #     export_code.export_basic_resNet_script(responses, best_params, basic_temp)
                # if arch_choice == '2':
                #     export_code.export_basic_vgg16_script(responses, best_params, basic_temp)
                # if arch_choice == '3':
                #     export_code.export_basic_leNet_Script(responses, best_params, basic_temp)
                # if arch_choice == '4':
                #     export_code.export_basic_alexNet_Script(responses, best_params, basic_temp)
                return best_params
            else:

                # Return a message or handle the logic when hyperparameter tuning is not selected
                return {"message": "Hyperparameter tuning not selected."}

        except ValueError as e:
            # Handle case where conversion to int fails
            return {
                "error": "Invalid input for image size, num classes, rgb_or_grey, hyp_choice, arch_choice, or n_trials. Must be integers."}

        except Exception as e:
            # Catch all other exceptions
            return {"error": f"An error occurred: {str(e)}"}
    def hello_World(self):
        print("Hello")
        return 'hello'


    def choose_hyperparameter_optimization_method(self, responses, n_trials):
        choice = responses[4]
        print("_________________ inside choose_hyperparameter_optimization_method ____________________")

        if choice == 1:
            print("_________________ choice 1 for choose_hyperparameter_optimization_method ____________________")
            train_dataset, val_dataset, _ = self.load_my_dataset(responses)
            # n_trials = self.get_valid_input("Enter the number of trials for Optuna: ",
            #                                 lambda x: x.isdigit(),
            #                                 "Please enter a valid number.")
            n_trials = int(n_trials)
            return self.setup_optuna(n_trials, responses, train_dataset, val_dataset)

        elif choice == 2:
            print("_________________ choice 2 for choose_hyperparameter_optimization_method ____________________")
            train_dataset, val_dataset, _ = self.load_my_dataset(responses)
            # n_trials = self.get_valid_input("Enter the number of trials for Random Search: ",
            #                                 lambda x: x.isdigit(),
            #                                 "Please enter a valid number.")
            # n_trials = int(n_trials)
            return self.setup_grid_search(responses, train_dataset, val_dataset)

        elif choice == 3:
            print("_________________ choice 3 for choose_hyperparameter_optimization_method ____________________")
            train_dataset, val_dataset, _ = self.load_my_dataset(responses)
            # n_trials = self.get_valid_input("Enter the number of trials for Random Search: ",
            #                                 lambda x: x.isdigit(),
            #                                 "Please enter a valid number.")
            n_trials = int(n_trials)
            return self.setup_random_search(responses,n_trials, train_dataset, val_dataset)

        elif choice == 4:
            return self.setup_bayesian_optimization()
        else:
            print("Invalid choice. Please select a valid option.")
            return None

    def setup_optuna(self, n_trials, responses, train_dataset, val_dataset):
        print("_________________ inside setup_optuna ____________________")
        def objective_wrapper(trial):
            return self.objective(trial, responses, train_dataset, val_dataset)  # Ensure this method accepts a trial and operates correctly with it

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_wrapper, n_trials=n_trials)  # Adjust n_trials as needed

        # Now, you can access the best trial results
        best_params = study.best_trial.params
        print("Best hyperparameters found:", best_params)
        return best_params

    # Your existing Optuna setup code
    def objective(self, trial, responses, train_dataset, val_dataset):
        # responses = [int(image_size), int(num_classes), dataset_path, rgb_or_grey, suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Example: Define hyperparameters using trial suggestions
        # train_dataset, val_dataset, _ = self.load_my_dataset(responses)
        print("_________________ inside objective method ____________________")
        image_size = int(responses[0])
        NUM_CLASSES = int(responses[1])   #adjust based on your dataset
        dataset_path = responses[2]

        filters = trial.suggest_categorical('filters', [16, 32, 64, 128])
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)


        model = self.choose_cnn_architecture(responses, filters,dropout_rate, learning_rate)

        # model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
        #               metrics=['accuracy'])

        # Fit the model
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=0)
        # Assuming X_val and y_val are your validation dataset and labels
        val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
        # if val_accuracy > best_accuracy:
        #     best_accuracy = val_accuracy
        #     print(f"New best accuracy: {best_accuracy}")
        return val_accuracy

    # def objective(self, trial, responses, train_dataset, val_dataset):
    #     # Load and preprocess CIFAR-10 dataset
    #     (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    #     train_images, test_images = train_images / 255.0, test_images / 255.0
    #     train_labels, test_labels = to_categorical(train_labels, 10), to_categorical(test_labels, 10)
    #
    #     # Suggest hyperparameters
    #     filters = trial.suggest_categorical('filters', [16, 32, 64, 128])
    #     dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    #     learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    #
    #     # Choose CNN architecture based on trial's suggested parameters
    #     # Here you need to modify choose_cnn_architecture to accept and utilize the suggested parameters
    #     model = self.choose_cnn_architecture(responses,filters, dropout_rate, learning_rate)
    #
    #     # Compile the model
    #     model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
    #                   metrics=['accuracy'])
    #
    #     # Fit the model on training data
    #     history = model.fit(train_images, train_labels, epochs=1, validation_split=0.1, verbose=0)
    #
    #     # Evaluate the model on the test set
    #     _, val_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    #
    #     # Optionally, you might want to print the val_accuracy to monitor the optimization process
    #     print(f"Validation Accuracy: {val_accuracy}")
    #
    #     return val_accuracy


    def setup_grid_search(self, responses, train_dataset, val_dataset):
        # Define the grid of hyperparameters to search through
        filter_options = [16, 32, 64, 128]
        dropout_rate_options = np.linspace(0.0, 0.5, num=5)  # 5 values evenly spaced between 0.0 and 0.5
        learning_rate_options = [1e-5, 1e-4, 1e-3, 1e-2]

        best_accuracy = 0
        best_hyperparameters = {}

        # Iterate over every combination of hyperparameters
        for filters in filter_options:
            for dropout_rate in dropout_rate_options:
                for learning_rate in learning_rate_options:
                    # Build and compile the model with current set of hyperparameters
                    model = self.choose_cnn_architecture(responses, filters=filters, dropout_rate=dropout_rate)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])

                    # Fit the model and evaluate
                    model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=0)
                    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
                    print(f"Best accuracy: {best_accuracy}")
                    print(f"Best hyperparameters: {best_hyperparameters}")

                    # Update the best hyperparameters if the current trial is better
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_hyperparameters = {
                            'filters': filters,
                            'dropout_rate': dropout_rate,
                            'learning_rate': learning_rate
                        }

        print(f"Best accuracy: {best_accuracy}")
        print(f"Best hyperparameters: {best_hyperparameters}")
        return best_hyperparameters

    def setup_random_search(self, responses, n_trials, train_dataset, val_dataset):
        # Define the search space for hyperparameters
        filter_options = [16, 32, 64, 128]
        dropout_rate_options = np.linspace(0.0, 0.5, num=10)  # 10 linearly spaced values
        learning_rate_options = [1e-5, 1e-4, 1e-3, 1e-2]

        best_accuracy = 0
        best_hyperparameters = {}

        for _ in range(n_trials):
            # Randomly select hyperparameters
            filters = random.choice(filter_options)
            dropout_rate = random.choice(dropout_rate_options)
            learning_rate = random.choice(learning_rate_options)

            # Build and compile the model with chosen hyperparameters
            # Here, choose_cnn_architecture needs to accept filters, dropout_rate as arguments
            model = self.choose_cnn_architecture(responses,filters, dropout_rate, learning_rate)


            # Fit the model and evaluate
            model.fit(train_dataset, validation_data=val_dataset, epochs=10, verbose=0)
            val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
            print(f"Best accuracy: {best_accuracy}")
            print(f"Best hyperparameters: {best_hyperparameters}")

            # Update the best hyperparameters if current trial is better
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_hyperparameters = {'filters': filters, 'dropout_rate': dropout_rate,
                                        'learning_rate': learning_rate}
            print(f"Best accuracy: {best_accuracy}")
            print(f"Best hyperparameters: {best_hyperparameters}")

        print(f"Best accuracy: {best_accuracy}")
        print(f"Best hyperparameters: {best_hyperparameters}")
        return best_hyperparameters

    def setup_bayesian_optimization(self):
        # Define or prompt for search spaces
        # Implement the method for Bayesian Optimization
        print("Bayesian Optimization setup is not implemented yet.")





    def load_my_dataset(self, responses):
        print("_________________ inside load_my_dataset method ____________________")
        # responses = [int(image_size), int(num_classes), dataset_path, rgb_or_grey, suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Assuming dataset_path is structured with /train, /test, and /val subdirectories
        train_dir = f'{responses[2]}/train'
        test_dir = f'{responses[2]}/test'
        val_dir = f'{responses[2]}/validation'

        # Define the normalization layer
        normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)

        # Load datasets from directories and apply normalization
        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            label_mode='categorical',  # Use 'categorical' for multi-class labels
            image_size=(responses[0], responses[0]),
            batch_size=32  # A common batch size choice
        ).map(lambda x, y: (normalization_layer(x), y))  # Apply normalization

        test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            test_dir,
            label_mode='categorical',
            image_size=(responses[0], responses[0]),
            batch_size=32
        ).map(lambda x, y: (normalization_layer(x), y))  # Apply normalization

        val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            label_mode='categorical',
            image_size=(responses[0], responses[0]),
            batch_size=32
        ).map(lambda x, y: (normalization_layer(x), y))  # Apply normalization

        return (train_dataset, test_dataset, val_dataset)

    from tensorflow.keras import layers, models

    def choose_cnn_architecture(self, responses, filters=64, dropout_rate=0.5, learning_rate=1e-3):
        print("_________________ inside choose_cnn_architecture method ____________________")
        def is_valid_choice(choice):
            return choice in [1, 2, 3, 4]

        def is_valid_complexity(complexity):
            return complexity.lower() in ['high', 'mid', 'low']
        choice = responses[5]
        if is_valid_choice(choice):
            if choice == 1:
                print("_________________ inside choice 1 in choose_cnn_architecture method ____________________")
                return self.build_resNet(responses, learning_rate=learning_rate)
            elif choice == 2:
                print("_________________ inside choice 2 in choose_cnn_architecture method ____________________")
                return self.build_vgg(responses, filters=filters, dropout_rate=dropout_rate, learning_rate=learning_rate)

            elif choice == 3:
                print("_________________ inside choice 3 in choose_cnn_architecture method ____________________")
                return self.build_leNet(responses, learning_rate=learning_rate)
            elif choice == 4:
                print("_________________ inside choice 4 in choose_cnn_architecture method ____________________")
                return self.build_alex_net(responses, dropout_rate=dropout_rate, learning_rate=learning_rate)
        else:
            print("Invalid choice. Please select a valid option.")

    def build_vgg(self, responses, filters=64, dropout_rate=0.5, learning_rate=1e-3):
        # responses = [int(image_size), int(num_classes), dataset_path, int(rgb_or_grey), suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Assuming responses[0] contains the image size and your images are RGB
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
        input_shape = (img_size, img_size, channels)  # (height, width, channels)

        model = tf.keras.models.Sequential([
            # Input layer
            tf.keras.layers.InputLayer(input_shape=input_shape),

            # Block 1
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 2
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

            # Block 3
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

            # Block 4
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

            # Block 5
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),

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
    def build_alex_net(self, responses, dropout_rate=0.5, learning_rate=1e-3):
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
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

        # Compile the model with the specified learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def build_leNet(self, responses, learning_rate=1e-3):
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
        input_shape = (img_size, img_size, channels)  # (height, width, channels)
        # using lenet architecture
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

        # Compile the model with the specified learning rate
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

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

    def build_resNet(self, responses, learning_rate=1e-3):
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]  # Directly using the integer value from responses[3] for channels
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

    def adjust_conv_layers_based_on_image_size(self, image_size):
        # This is a simplistic approach; you might want to use more sophisticated logic
        layers = []
        if image_size > 200:
            # Larger images, more layers
            layers.append({'filters': 64, 'kernel_size': (3, 3)})
            layers.append({'filters': 128, 'kernel_size': (3, 3)})
            layers.append({'filters': 256, 'kernel_size': (3, 3)})
        elif image_size > 100:
            # Medium images, moderate layers
            layers.append({'filters': 32, 'kernel_size': (3, 3)})
            layers.append({'filters': 64, 'kernel_size': (3, 3)})
        else:
            # Smaller images, fewer layers
            layers.append({'filters': 32, 'kernel_size': (3, 3)})
        return layers

    def generate_image_classification_cnn(self, responses,best_params):
        image_size = int(responses[0])  # Assuming the first response is the image size
        suggest_dataset = 'n'
        use_data_augmentation = 'n'
        NUM_CLASSES = int(responses[1])  # number of categories or classes
        dataset_path = responses[2]
        conv_layers = self.adjust_conv_layers_based_on_image_size(image_size)

        filters = best_params.get('filters', 32)  # Default to 32 if not specified
        dropout_rate = best_params.get('dropout_rate', 0.5)  # Default value if not specified
        learning_rate = best_params.get('learning_rate', 1e-3)  # Default value if not specified

        data_augmentation_str = """# no selected data augmentation"""
        suggest_dataset_str = """"""

        if use_data_augmentation == 'y':
            data_augmentation_str = """
        # Data augmentation
        data_augmentation = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )"""
        else:
            data_augmentation_str

        if suggest_dataset == 'y':
            suggest_dataset_str = "#Consider using the CIFAR-10 dataset for a basic image classification task."
        else:
            suggest_dataset_str

        model_layers = []
        for i, layer in enumerate(conv_layers):
            model_layers.append(
                f"Conv2D(filters={layer['filters']}, kernel_size={layer['kernel_size']}, activation='relu')")
            # Add a MaxPooling2D layer after each Conv2D layer
            if i < len(conv_layers) - 1:  # Avoid adding pooling after the last conv layer
                model_layers.append("MaxPooling2D(pool_size=(2, 2))")

        model_layers_str = ',\n    '.join(model_layers)

        skeleton = f"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        #define the paths to dataset
        {dataset_path}
        
    
        model = Sequential([
            {model_layers_str},
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense({NUM_CLASSES}, activation='softmax')
        ])
    
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        {data_augmentation_str}
        # Example of how to use data augmentation (if enabled):
        # model.fit(data_augmentation.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))
        
        {suggest_dataset_str}
        """
        print("Image Classification CNN Skeleton:\n", skeleton)

        filename = "generated_cnn_model.py"
        # Write the skeleton to the file
        with open(filename, 'w') as file:
            file.write(skeleton)
        print(f"\nGenerated code has been saved to {filename}")
