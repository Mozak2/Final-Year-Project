from app.codingAssistant import CodingAssistant

coding_assistant = CodingAssistant()
coding_assistant.present_questions()

selected_option = input("Enter the number of your choice: ")
coding_assistant.generate_code(selected_option)
