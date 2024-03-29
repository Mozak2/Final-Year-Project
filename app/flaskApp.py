from flask import Flask, request, render_template, jsonify


# from app.codingAssistant import CodingAssistant

# Ensure you have imported your CodingAssistant correctly

app = Flask(__name__)
from codingAssistant import CodingAssistant
from export_templates import export_code
coding_assistant = CodingAssistant()
# export_template = export_code()
# A simple route that renders a welcome message
@app.route('/customize', methods=['GET', 'POST'])
def customize_model():
    if request.method == 'POST':
        try:
            basic_temp= 'basic template.py'
            # Extracting form datasgsdrgdsg
            image_size = request.form['image_size']
            num_classes = request.form['num_classes']
            rgb_or_grey = request.form['rgb_or_grey']
            dataset_path = request.form['dataset_path']
            hyperparam_tuning = request.form['hyperparam_tuning']
            hyp_choice = request.form.get('hyp_choice', '0')  # Default to '1' if not provided
            arch_choice = request.form['arch_choice']
            n_trials = request.form['n_trials']
            # Process the extracted data with your coding assistant
            print("_________________ console log 1 ____________________")
            responses = [image_size,num_classes , dataset_path, rgb_or_grey, hyp_choice, arch_choice, n_trials]
            print("_________________ console log 2 ____________________")
            # best_params = coding_assistant.customise_image_classification_param(image_size, num_classes, rgb_or_grey, dataset_path,
            #                                                       hyperparam_tuning, hyp_choice, arch_choice, n_trials)
            #
            # print("_________________ console log 3 lol ____________________")
            # print("Best hyperparameters: ", best_params)


            # coding_assistant.hello_World()
            # if arch_choice == '1':
            #     result = export_template.export_basic_resNet_script(responses, best_params, basic_temp)
            # if arch_choice == '2':
            #     result = export_template.export_basic_vgg16_script(responses, best_params, basic_temp)
            # if arch_choice == '3':
            #     result = export_template.export_basic_leNet_Script(responses, best_params, basic_temp)
            # if arch_choice == '4':
            #     result = export_template.export_basic_alexNet_Script(responses, best_params, basic_temp)

            file_path = 'C:/Users/moiib/PycharmProjects/Final-Year-Project/basic template.py'
            with open(file_path, 'r') as file:
                code_content = file.read()

                # Render the results template with the code
            return render_template('result.html', code=code_content)
            # After processing, you might want to redirect or render a template
            # return jsonify({"success": True, "message": "Parameters customized successfully."})

        except KeyError as e:
            # Handle missing form data
            return jsonify({"error": f"Missing data: {str(e)}"}), 400

        except Exception as e:
            # Handle other errors
            return jsonify({"error": str(e)}), 500

    else:  # GET request
        # Render the form for the user to fill in
        return render_template('index.html')  # Assume your HTML form is saved as form.html


if __name__ == "__main__":
    app.run(debug=True)
