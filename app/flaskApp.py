from flask import Flask, render_template, request
from app.codingAssistant import CodingAssistant

app = Flask(__name__)
coding_assistant = CodingAssistant()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_option = request.form['selected_option']
        generated_code = coding_assistant.generate_code(selected_option)
        return render_template('index.html', generated_code=generated_code)
    else:
        coding_assistant.present_questions()
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
