class QuestionRepository:
    def __init__(self):
        self.questions = {
            "1. Do you want to create a CNN for image classification?":
                "This question is about building a Convolutional Neural Network (CNN) to classify images into different categories.",
            "2. Do you want to build a CNN for object detection?":
                "This question is about constructing a Convolutional Neural Network (CNN) to detect objects within images.",
            "3. Do you want to train a CNN for facial recognition?":
                "This question is about training a Convolutional Neural Network (CNN) to recognize faces in images."
        }
        self.facialquestions = {
            "1. Do you want to create a CNN for image classification?":
                "This question is about building a Convolutional Neural Network (CNN) to classify images into different categories.",
            "2. Do you want to build a CNN for object detection?":
                "This question is about constructing a Convolutional Neural Network (CNN) to detect objects within images.",
            "3. Do you want to train a CNN for facial recognition?":
                "This question is about training a Convolutional Neural Network (CNN) to recognize faces in images."
        }
        self.objectquestions = {
            "1. Do you want to create a CNN for image classification?":
                "This question is about building a Convolutional Neural Network (CNN) to classify images into different categories.",
            "2. Do you want to build a CNN for object detection?":
                "This question is about constructing a Convolutional Neural Network (CNN) to detect objects within images.",
            "3. Do you want to train a CNN for facial recognition?":
                "This question is about training a Convolutional Neural Network (CNN) to recognize faces in images."
        }
        self.imagequestions = {
            "1. Do you want to create a CNN for image classification?":
                "This question is about building a Convolutional Neural Network (CNN) to classify images into different categories.",
            "2. Do you want to build a CNN for object detection?":
                "This question is about constructing a Convolutional Neural Network (CNN) to detect objects within images.",
            "3. Do you want to train a CNN for facial recognition?":
                "This question is about training a Convolutional Neural Network (CNN) to recognize faces in images."
        }
    def get_questions(self):
        return self.questions
