import optuna

from app.questionsRepository import QuestionRepository
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from flask import Flask, render_template, request


class CodingAssistant:
    def __init__(self):
        self.question_repo = QuestionRepository()

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

        hyperparam_tuning = self.get_valid_input("Would you like your hyperparameters to be tuned? (y/n): ",
                                                 lambda x: x.lower() in ['y', 'n'])
        if hyperparam_tuning == 'y':
            def objective_wrapper(trial):
                return self.objective(trial)  # Ensure this method accepts a trial and operates correctly with it

            study = optuna.create_study(direction='maximize')
            study.optimize(objective_wrapper, n_trials=10)  # Adjust n_trials as needed

            # Now, you can access the best trial results
            best_params = study.best_trial.params
            print("Best hyperparameters found:", best_params)

        image_size = self.get_valid_input("Enter image size (e.g., 128 for 128x128 pixels): ",
                                          lambda x: x.isdigit() and int(x) > 0)

        suggest_dataset = self.get_valid_input("Do you need a suggestion for a dataset? (y/n): ",
                                               lambda x: x.lower() in ['y', 'n'])

        use_data_augmentation = self.get_valid_input("Would you like to use data augmentation? (y/n): ",
                                                     lambda x: x.lower() in ['y', 'n'])

        num_classes = self.get_valid_input("How many classes would you like? (e.g., 10): ",
                                           lambda x: x.isdigit() and int(x) > 0)
        # Prompt user for the path to their dataset
        dataset_path = self.get_valid_input(
            "Enter the path to your dataset (including 'Dataset =') or leave blank if not applicable): ",
            lambda x: True)  # Validation allows any input, including blank for flexibility
        responses = [int(image_size), suggest_dataset.lower(), use_data_augmentation.lower(), int(num_classes),
                     dataset_path]

        self.generate_image_classification_cnn(responses)

    def choose_hyperparameter_optimization_method(self):
        print("Select a hyperparameter optimization method:")
        print("1: Optuna")
        print("2: Grid Search")
        print("3: Random Search")
        print("4: Bayesian Optimization")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            def objective_wrapper(trial):
                return self.objective(trial)  # Ensure this method accepts a trial and operates correctly with it

            study = optuna.create_study(direction='maximize')
            study.optimize(objective_wrapper, n_trials=10)  # Adjust n_trials as needed

            # Now, you can access the best trial results
            best_params = study.best_trial.params
            print("Best hyperparameters found:", best_params)
        elif choice == '2':
            self.setup_grid_search()
        elif choice == '3':
            self.setup_random_search()
        elif choice == '4':
            self.setup_bayesian_optimization()
        else:
            print("Invalid choice. Please select a valid option.")

    def setup_grid_search(self):
        # Define or prompt for search spaces
        # Implement the method for grid search
        print("Grid Search setup is not implemented yet.")

    def setup_random_search(self):
        # Define or prompt for search spaces
        # Implement the method for random search
        print("Random Search setup is not implemented yet.")

    def setup_bayesian_optimization(self):
        # Define or prompt for search spaces
        # Implement the method for Bayesian Optimization
        print("Bayesian Optimization setup is not implemented yet.")

    def objective(self, trial):
        # Example: Define hyperparameters using trial suggestions
        # image_size = trial.suggest_categorical('image_size', [32, 64, 128, 256])
        image_size = 32  # Fixed size for CIFAR-10 images
        filters = trial.suggest_categorical('filters', [16, 32, 64])
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        NUM_CLASSES = 10  # Example, adjust based on your dataset

        # Assuming X and y are your full dataset and labels

        # Ask the user for the dataset path
        #     dataset_path = input("Please enter the path to your dataset: ")

        # Load the dataset based on the provided path
        # This is a placeholder, replace with actual code to load your dataset
        #     X, y = load_dataset(dataset_path)

        # Load the dataset
        (X, y), (X_test, y_test) = cifar10.load_data()

        # Normalize pixel values to be between 0 and 1
        X, X_test = X / 255.0, X_test / 255.0


        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25,
                                                          random_state=42)  # 0.25 * 0.8 = 0.2

        # If your labels are not already one-hot encoded
        y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
        y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
        # Your model building logic here, using the above hyperparameters
        model = Sequential([
            Conv2D(filters=filters, kernel_size=(3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(NUM_CLASSES, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Fit the model (consider using a subset of your data to speed up the process)
        # Evaluate the model to return the metric of interest, e.g., validation accuracy
        # Assuming X_val and y_val are your validation dataset and labels
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
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

    def generate_image_classification_cnn(self, responses):
        image_size = int(responses[0])  # Assuming the first response is the image size
        suggest_dataset = responses[1].lower()
        use_data_augmentation = responses[2].lower()
        NUM_CLASSES = int(responses[3])  # number of categories or classes
        dataset_path = responses[4]
        img_size = responses[5]
        dropout_rate = responses[6]
        learning_rate = responses[7]
        conv_layers = self.adjust_conv_layers_based_on_image_size(image_size)

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
