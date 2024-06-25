from flask import Flask, render_template, request, flash

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from joblib import load

# import uuid

import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Necessary for flash messages.

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    request_type = request.method
    if request_type == "GET":
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        
        if not text:
            flash('Input is required!', 'error')
            return render_template('index.html', href='static/base_pic.svg')
        
        try:
            input_np_array = floats_string_to_np_array(text)

        except ValueError as e:
            flash(str(e), 'error')
            return render_template('index.html', href='static/base_pic.svg')
        
        # file_name = uuid.uuid4().hex
        
        # path = 'static/' + file_name + '.svg'
        path = 'static/preds_img.svg'
        model = load('model.joblib')
        input_np_array = floats_string_to_np_array(text)
        make_picture('AgesAndHeights.pkl', model, input_np_array, path)
        return render_template('index.html', href=path)


def make_picture(training_data_filename, model, new_input, output_file):
    data = pd.read_pickle(training_data_filename)
    ages = data['Age']
    heights = data['Height']
    data = data[ages > 0]
    x_new = np.array(list(range(19))).reshape(-1, 1)
    preds = model.predict(x_new)

    fig = px.scatter(x=ages, 
                 y=heights,
                 title="Heights vs Age of People", 
                 labels={'x': 'Age (years)', 
                         'y': 'Height (inches)'})
    fig.add_trace(go.Scatter(x=x_new.reshape(-1,), y=preds, mode='lines', name='Model'))

    new_preds = model.predict(new_input)

    fig.add_trace(go.Scatter(x=new_input.reshape(-1), 
                             y=new_preds, 
                             name="New Outputs", 
                             mode="markers", 
                             marker=dict(color='purple', size=20, line=dict(color='purple', width=2))))

    fig.write_image(output_file, width=800, engine='kaleido')
    # fig.show()


# def floats_string_to_np_array(floats_str):
#     def is_float(x):
#         try:
#             float(x)
#             return True
#         except:
#             return False
#     floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)]) 
#     return floats.reshape(-1, 1)

def floats_string_to_np_array(floats_str):
    try:
        # Split the string by commas and strip whitespace from each part
        # Filter out any empty strings that result from trailing commas or extra spaces
        floats_list = [float(item.strip()) for item in floats_str.split(',') if item.strip()]
        return np.array(floats_list).reshape(-1, 1)
    except ValueError:
        raise ValueError("Invalid input! Please enter comma-separated numbers.")


# if __name__ == "__main__":
#     app.run(debug=True)
