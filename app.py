from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import sqlite3
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

# Route to the home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict_route():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', filename)
            file.save(filepath)

            # Load and preprocess the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            predicted_class, confidence = predict(img)

            return render_template('predict.html', image_path=filename, predicted_label=predicted_class, confidence=confidence)

        return render_template('predict.html', message='Invalid file type. Please upload a JPG/PNG image.')
    
    return render_template('predict.html')

# Crop search route with dropdown menu
@app.route('/search', methods=['GET', 'POST'])
def search_crop():
    crop_list = ['Potato', 'Corn', 'Rice', 'Wheat', 'Sugarcane']
    selected_crop = None
    crop_info = None
    
    if request.method == 'POST':
        selected_crop = request.form.get('crop')
        conn = sqlite3.connect('crops.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM crops WHERE name = ?", (selected_crop,))
        crop_info = cursor.fetchone()
        conn.close()
    
    return render_template('search.html', crop_list=crop_list, selected_crop=selected_crop, crop_info=crop_info)

if __name__ == '__main__':
    app.run(debug=True)