from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, storage
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from numpy.linalg import norm
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
import pickle
import tensorflow as tf
import logging
from flask_cors import CORS

logging.basicConfig(filename='app.log', level=logging.INFO)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './uploads'

cred = credentials.Certificate("firebase connection key")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'firebase storage'
})
bucket = storage.bucket()

feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filename_list = pickle.load(open("filename.pkl", "rb"))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

def get_recommendations(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([features])
    recommended_filenames = [filename_list[i] for i in indices[0][1:]]  # Excludes the query image
    return recommended_filenames

@app.route('/recommend', methods=['POST'])
def recommend_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(file_path)

    features = extract_feature(file_path, model)
    recommended_filenames = get_recommendations(features, feature_list)
    logging.info(f"Recommended filenames: {recommended_filenames}")

    recommended_image_urls = []
    for recommended_filename in recommended_filenames:
        local_filename = recommended_filename.replace('Dataset/', '')
        src_path = os.path.join('Dataset', local_filename)

        try:
            blob = bucket.blob(f'recommend/{local_filename}')
            blob.upload_from_filename(src_path)
            recommended_image_urls.append(blob.public_url)
        except Exception as e:
            logging.error(f"Failed to upload file: {e}")

    return jsonify(recommended_image_urls)

@app.route('/delete_image', methods=['POST'])
def delete_image():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    try:
        blob = bucket.blob(f'recommend/{filename}')
        blob.delete()
        return jsonify({"message": "Image deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
