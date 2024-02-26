from app.questionsRepository import QuestionRepository
import tensorflow as tf
from tensorflow.keras import layers, models
from flask import Flask, render_template, request

class CodingAssistant:
    def __init__(self):
        self.question_repo = QuestionRepository()

    def start_interaction(self):
        general_questions = self.question_repo.get_general_questions()
        print("Please select an option:")
        for idx, question in enumerate(general_questions, 1):
            print(f"{idx}. {question}")
        choice = int(input("Enter the number of your choice: "))  # Capture the user's choice
        # print(f"Choice: {choice}, Type: {type(choice)}")
        self.generate_code_based_on_choice(choice)  # Adjusted to call the method with the correct index

    def generate_code_based_on_choice(self, choice):
        # print(f"Choice: {choice}, Type: {type(choice)}")
        if choice == 1:  # Assuming 1 corresponds to image classification, for example
            self.customise_image_classification_param()
        elif choice == 2:
            return self.generate_object_detection_cnn()
        elif choice == 3:
            return self.generate_facial_recognition_cnn()
        else:
            return "Invalid option selected."

        # Include similar conditions for other types of CNN projects

    # def customise_image_classification_param(self):
    #     # Assuming no need for dynamic responses collection for now
    #     # Prompt user for image size
    #     image_size = input("Enter image size (e.g., 128 for 128x128 pixels): ")
    #     # Ask user if they want dataset suggestions
    #     suggest_dataset = input("Do you need a suggestion for a dataset? (yes/no): ").lower()
    #     # Ask user if they want to use data augmentation
    #     use_data_augmentation = input("Would you like to use data augmentation? (yes/no): ").lower()
    #     # Ask user the number of classes/categories they want
    #     num_classes = input("How many Classes would you like? (how many categories for final layer of CNN. e.g. 10 ): ")
    #
    #
    #     # Collect responses
    #     responses = [image_size, suggest_dataset, use_data_augmentation, num_classes]
    #     self.generate_image_classification_cnn(responses)

    def get_valid_input(self, prompt, validation_func, error_message="Invalid answer. Please try again."):
        while True:
            user_input = input(prompt)
            if validation_func(user_input):
                return user_input
            print(error_message)

    def customise_image_classification_param(self):
        image_size = self.get_valid_input("Enter image size (e.g., 128 for 128x128 pixels): ",
                                          lambda x: x.isdigit() and int(x) > 0)

        suggest_dataset = self.get_valid_input("Do you need a suggestion for a dataset? (y/n): ",
                                               lambda x: x.lower() in ['y', 'n'])

        use_data_augmentation = self.get_valid_input("Would you like to use data augmentation? (y/n): ",
                                                     lambda x: x.lower() in ['y', 'n'])

        num_classes = self.get_valid_input("How many classes would you like? (e.g., 10): ",
                                           lambda x: x.isdigit() and int(x) > 0)
        # Prompt user for the path to their dataset
        dataset_path = self.get_valid_input("Enter the path to your dataset (including Dataset =) or leave blank if not applicable): ",
                                            lambda x: True)  # Validation allows any input, including blank for flexibility

        responses = [int(image_size), suggest_dataset.lower(), use_data_augmentation.lower(), int(num_classes), dataset_path]
        self.generate_image_classification_cnn(responses)

    # def collect_follow_up_responses(self, category):
    #     follow_up_questions = self.question_repo.get_follow_up_questions(category)
    #     responses = []
    #     print(f"\nFollow-up Questions for {category}:")
    #     for question in follow_up_questions:
    #         print(question)
    #         response = input("Your answer: ")
    #         responses.append(response)
    #     return responses

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
        NUM_CLASSES = int(responses[3]) # number of categories or classes
        dataset_path = responses[4]
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
        else: data_augmentation_str

        if suggest_dataset == 'y':
            suggest_dataset_str = "#Consider using the CIFAR-10 dataset for a basic image classification task."
        else: suggest_dataset_str

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

    def generate_object_detection_cnn(self):
        # Define a CNN model for object detection
        # Add code here for object detection model
        print("job not finished")
        return "this aint done yet"

    def generate_facial_recognition_cnn(self):
        # Define a CNN model for facial recognition
        # Add code here for facial recognition model
        return "Facial recognition CNN code generated."
