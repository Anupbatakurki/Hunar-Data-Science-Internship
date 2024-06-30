from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'path/to/your/saved_model.h5'  # Update with the correct path to your saved model
model = load_model(MODEL_PATH)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Prepare the image
        img = image.load_img(file_path, target_size=(150, 150))  # Update target size according to your model
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        if prediction[0][0] > 0.5:
            result = "It's a Dog"
        else:
            result = "It's a Cat"
        
        return render_template('result.html', prediction=result, image_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
