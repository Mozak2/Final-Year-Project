import optuna

from app.questionsRepository import QuestionRepository
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
        self.question_repo = QuestionRepository()
        self.architecture_choice = None
        self.complexity_choice = None

    def start_interaction(self):
        general_questions = self.question_repo.get_general_questions()
        print("Please select an option:")
        start = self.get_valid_input("Would you like to get started? (y/n): ",
                                                 lambda x: x.lower() in ['y', 'n'])
        if start == 'y':
            self.customise_image_classification_param();


    def get_valid_input(self, prompt, validation_func, error_message="Invalid answer. Please try again."):
        while True:
            user_input = input(prompt)
            if validation_func(user_input):
                return user_input
            print(error_message)

    def customise_image_classification_param(self):

        # hyperparam_tuning = 0
        # image_size = self.get_valid_input("Enter image size (e.g., 128 for 128x128 pixels): ",
        #                                       lambda x: x.isdigit() and int(x) > 0)
        #
        # num_classes = self.get_valid_input("How many classes would you like? (e.g., 10): ",
        #                                    lambda x: x.isdigit() and int(x) > 0)
        #
        # rgb_or_grey = self.get_valid_input("Do you want to use rgb or greyscale? (rgb = 3 / greyscale = 1) ",
        #                                    lambda x: x.isdigit() and int(x) in [1, 3])
        #
        # #  suggest_dataset = self.get_valid_input("Do you need a suggestion for a dataset? (y/n): ",
        # #                                        lambda x: x.lower() in ['y', 'n'])
        # #
        # # use_data_augmentation = self.get_valid_input("Would you like to use data augmentation? (y/n): ",
        # #                                              lambda x: x.lower() in ['y', 'n'])
        #
        #
        # # Prompt user for the path to their dataset
        # dataset_path = self.get_valid_input(
        #     "Enter the path to your dataset or leave blank if not applicable): ",
        #     lambda x: True)  # Validation allows any input, including blank for flexibility

        # responses = [int(image_size), int(num_classes), dataset_path, rgb_or_grey]

        responses = [32, 3, 'C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)',3]
        # C:/Users/moiib/PycharmProjects/Final-Year-Project/archive (1)
        # hyperparam_tuning = self.get_valid_input("Would you like your hyperparameters to be tuned? (y/n): ",
        #                                          lambda x: x.lower() in ['y', 'n'])
        print("Select a CNN architecture:")
        print("1: ResNet")
        print("2: VGG")
        print("3: Basic CNN")
        print("4: EfficientNet")
        choice = input("Enter your choice (1-4): ")

        hyperparam_tuning = 'y'
        if hyperparam_tuning == 'y':
            hyp_choice = self.get_valid_input("Select a hyperparameter optimization method:\n1: Optuna\n2: Grid Search"
                                              "\n3: Random Search\n4: Bayesian Optimization\nEnter your choice (1-4): ",
                                              lambda x: x.isdigit() and int(x) in [1, 2, 3, 4])
            # arch_choice = self.get_valid_input("Select a hyperparameter optimization method:\n1: Optuna\n2: Grid Search"
            #                                   "\n3: Random Search\n4: Bayesian Optimization\nEnter your choice (1-4): ",
            #                                   lambda x: x.isdigit() and int(x) in [1, 2, 3, 4])

            responses.append(int(hyp_choice))

        best_params = self.choose_hyperparameter_optimization_method(responses)

        self.generate_image_classification_cnn(responses, best_params)

    def choose_hyperparameter_optimization_method(self, responses):
        choice = responses[4]
        if choice == 1:
            n_trials = self.get_valid_input("Enter the number of trials for Optuna: ",
                                            lambda x: x.isdigit(),
                                            "Please enter a valid number.")
            n_trials = int(n_trials)
            return self.setup_optuna(n_trials, responses)
        elif choice == '2':
            return self.setup_grid_search()
        elif choice == '3':
            n_trials = self.get_valid_input("Enter the number of trials for Random Search: ",
                                            lambda x: x.isdigit(),
                                            "Please enter a valid number.")
            n_trials = int(n_trials)
            return self.setup_random_search(responses,n_trials)
        elif choice == '4':
            return self.setup_bayesian_optimization()
        else:
            print("Invalid choice. Please select a valid option.")
            return None

    def setup_optuna(self, n_trials, responses):
        def objective_wrapper(trial):
            return self.objective(trial, responses)  # Ensure this method accepts a trial and operates correctly with it

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_wrapper, n_trials=n_trials)  # Adjust n_trials as needed

        # Now, you can access the best trial results
        best_params = study.best_trial.params
        # print("Best hyperparameters found:", best_params)
        return best_params

    # Your existing Optuna setup code
    # Ensure this method returns best_params or None if it fails
    def setup_grid_search(self):
        # Define or prompt for search spaces
        # Implement the method for grid search
        print("Grid Search setup is not implemented yet.")

    def setup_random_search(self, responses, n_trials):
        # Define the search space for hyperparameters
        # filter_options = [16, 32, 64, 128]
        # dropout_rate_options = np.linspace(0.0, 0.5, num=10)  # 10 linearly spaced values
        # learning_rate_options = [1e-5, 1e-4, 1e-3, 1e-2]

        best_accuracy = 0
        best_hyperparameters = {}

        for _ in range(n_trials):
            # Randomly select hyperparameters
            filters = random.choice([16, 32, 64, 128])
            dropout_rate = random.uniform(0.0, 0.5)
            learning_rate = random.choice([1e-5, 1e-4, 1e-3, 1e-2])

            # Load dataset
            train_dataset, val_dataset, _ = self.load_my_dataset(responses)

            # Build and compile the model with chosen hyperparameters
            # Here, choose_cnn_architecture needs to accept filters, dropout_rate as arguments
            model = self.choose_cnn_architecture(responses)
            model.compile(optimizer=Adam(learning_rate=learning_rate),
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

            # Fit the model and evaluate
            model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=0)
            val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)

            # Update the best hyperparameters if current trial is better
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_hyperparameters = {'filters': filters, 'dropout_rate': dropout_rate,
                                        'learning_rate': learning_rate}

        print(f"Best accuracy: {best_accuracy}")
        print(f"Best hyperparameters: {best_hyperparameters}")
        return best_hyperparameters

    def setup_bayesian_optimization(self):
        # Define or prompt for search spaces
        # Implement the method for Bayesian Optimization
        print("Bayesian Optimization setup is not implemented yet.")


    def load_my_dataset(self, responses):

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

    def choose_cnn_architecture(self, responses):
        def is_valid_choice(choice):
            return choice in ['1', '2', '3', '4']

        def is_valid_complexity(complexity):
            return complexity.lower() in ['high', 'mid', 'low']
        print("Select a CNN architecture:")
        print("1: ResNet")
        print("2: VGG")
        print("3: Basic CNN")
        print("4: EfficientNet")
        choice = input("Enter your choice (1-4): ")

        if is_valid_choice(choice):
            if choice == '1':

                return self.build_resnet(responses)
            elif choice == '2':
                return self.build_vgg(responses)
            elif choice == '3':
                while True:
                    while True:
                        complexity = input("Choose the complexity (high, mid, low): ").lower()
                        if is_valid_complexity(complexity):
                            return self.generate_cnn_architecture(responses, complexity)
                        else:
                            print("Invalid complexity. Please choose 'high', 'mid', or 'low'.")
            elif choice == '4':
                 return self.build_pretrained_efficientnet(responses)
        else:
            print("Invalid choice. Please select a valid option.")

    # def choose_cnn_architecture(self, responses):
    #     if self.architecture_choice is None:  # First time choice
    #         while True:
    #             print("Select a CNN architecture:")
    #             print("1: ResNet")
    #             print("2: VGG")
    #             print("3: Basic CNN")
    #             print("4: EfficientNet")
    #             choice = input("Enter your choice (1-4): ")
    #
    #             if choice == '1':
    #                 self.architecture_choice = self.build_resnet(responses)
    #                 break
    #             elif choice == '2':
    #                 self.architecture_choice = self.build_vgg(responses)
    #                 break
    #             elif choice == '3':
    #                 if self.complexity_choice is None:  # First time choosing complexity
    #                     while True:
    #                         complexity = input("Choose the complexity (high, mid, low): ").lower()
    #                         if complexity in ['high', 'mid', 'low']:
    #                             self.complexity_choice = complexity
    #                             break
    #                         else:
    #                             print("Invalid complexity. Please choose 'high', 'mid', or 'low'.")
    #                 self.architecture_choice = self.generate_cnn_architecture(responses, self.complexity_choice)
    #                 break
    #             elif choice == '4':
    #                 self.architecture_choice = self.build_pretrained_efficientnet(responses)
    #                 break
    #             else:
    #                 print("Invalid choice. Please select a valid option.")
    #
    #     return self.architecture_choice

    def build_vgg(self, responses):
        # responses = [int(image_size), int(num_classes), dataset_path, int(rgb_or_grey), suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Assuming responses[0] contains the image size and your images are RGB
        num_classes = responses[1]
        img_size = responses[0]
        channels = responses[3]
        input_shape = (img_size, img_size, channels)  # (height, width, channels)

        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),  # Define the input shape dynamically

            # Block 1
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 2
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 3
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Flattening and Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')  # Assuming num_classes is provided correctly
        ])

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

    def build_resnet(self, responses):
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
        return model

    def generate_cnn_architecture(self, responses, dataset_complexity):
        # responses = [int(image_size), int(num_classes), dataset_path, rgb_or_grey, suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        img_width = responses[0]
        img_height = responses[0]
        num_classes = responses[1]
        model = tf.keras.Sequential()

        # Base number of filters, based on perceived dataset complexity
        if dataset_complexity == 'low':
            base_filters = 16
            conv_blocks = 2
            dense_units = 64
        elif dataset_complexity == 'high':
            base_filters = 64
            conv_blocks = 4
            dense_units = 128
        else:  # medium
            base_filters = 32
            conv_blocks = 3
            dense_units = 128

        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=(img_height, img_width, 3)))

        # Convolutional blocks
        for i in range(conv_blocks):
            filters = base_filters * (2 ** i)
            model.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(tf.keras.layers.Flatten())

        # Fully connected layers
        model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
        if num_classes > 10:  # Assuming a more complex task requires more capacity
            model.add(tf.keras.layers.Dense(dense_units // 2, activation='relu'))

        # Output layer
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        return model

    def build_pretrained_efficientnet(self, responses):
        num_classes = int(responses[1])
        img_size = responses[0]
        channels = responses[3]  # Directly using the integer value from responses[3] for channels
        input_shape = (img_size, img_size, channels)
        # Load the pre-trained EfficientNet model, excluding its top (final) layer
        base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')

        # Freeze the layers of the base model to not train them again
        for layer in base_model.layers:
            layer.trainable = False

        # Add new top layers for your specific classification problem
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(1024, activation='relu')(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        # Construct the final model
        model = Model(inputs=base_model.input, outputs=outputs)

        return model

    def objective(self, trial, responses):
        # responses = [int(image_size), int(num_classes), dataset_path, rgb_or_grey, suggest_dataset.lower(),
        #              use_data_augmentation.lower()]
        # Example: Define hyperparameters using trial suggestions
        train_dataset, val_dataset, _ = self.load_my_dataset(responses)

        image_size = int(responses[0])
        NUM_CLASSES = int(responses[1])   #adjust based on your dataset
        dataset_path = responses[2]

        filters = trial.suggest_categorical('filters', [16, 32, 64, 128])
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)


        # model = self.build_vgg(responses)
        model = self.choose_cnn_architecture(responses)

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Fit the model
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=1, verbose=0)
        # Assuming X_val and y_val are your validation dataset and labels
        val_loss, val_accuracy = model.evaluate(val_dataset, verbose=0)
        return val_accuracy

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
