from flask import Flask, request, render_template

from app.codingAssistant import CodingAssistant

# Ensure you have imported your CodingAssistant correctly

app = Flask(__name__)

@app.route('/generate_cnn', methods=['POST'])
def generate_cnn():
    model_choice = int(request.form['model_choice'])
    # Common to all models
    image_size = request.form.get('image_size')
    suggest_dataset = request.form.get('suggest_dataset', 'n').lower() == 'y'
    use_data_augmentation = request.form.get('use_data_augmentation', 'n').lower() == 'y'
    num_classes = int(request.form.get('num_classes', 0))
    dataset_path = request.form.get('dataset_path', '')

    assistant = CodingAssistant()

    # Image Classification
    if model_choice == 1:
        code = assistant.generate_image_classification_cnn(image_size, suggest_dataset, use_data_augmentation, num_classes, dataset_path)
    # Extend this conditional block for other model types (2: Object Detection, 3: Facial Recognition) with appropriate parameters

    return render_template('index.html', code=code)

if __name__ == '__main__':
    app.run(debug=True)
