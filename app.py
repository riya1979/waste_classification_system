from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model.keras", compile=False)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        predicted_class = "Recyclable"
        confidence = round(prediction * 100, 2)
    else:
        predicted_class = "Organic"
        confidence = round((1 - prediction) * 100, 2)

    return predicted_class, confidence


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        

        prediction, confidence = predict_image(filepath)
        image_path = os.path.join('static', 'uploads', filename)

    return render_template(
        'classify.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)