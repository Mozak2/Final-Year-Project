class QuestionRepository:
    def __init__(self):
        self.questions = {
            "general": ["Do you want to create a CNN for image classification?"]
                        # "Do you want to build a CNN for object detection?",
                        # "Do you want to train a CNN for facial recognition?"]
            # "image_classification": ["Follow-up question 1 for image classification", "Follow-up question 2"],
            # "object_detection": ["Follow-up question 1 for object detection", "Follow-up question 2"],
            # "facial_recognition": ["Follow-up question 1 for facial recognition", "Follow-up question 2"],
        }

    def get_general_questions(self):
        return self.questions["general"]

    def get_follow_up_questions(self, category):
        if category in self.questions:
            return self.questions[category]
        else:
            return ["Invalid category"]
