from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.src.saving import load_model
import keras
import numpy as np

app = Flask(__name__)

modelo = load_model('ia_model/model_latest.keras')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        temp_file_path = 'temp/' + file.filename
        file.save(temp_file_path)

        image = keras.utils.load_img(temp_file_path, color_mode='grayscale', target_size=(256, 256))
        input_arr = keras.utils.img_to_array(image)
        input_arr = np.array([input_arr])

        predictions = modelo.predict(input_arr)

        result = "Es Original" if predictions > 0.5 else "Es Falso"
        
        return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)