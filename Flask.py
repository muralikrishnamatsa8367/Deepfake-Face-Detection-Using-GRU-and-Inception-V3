from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow import keras
import os 

app = Flask(__name__)

# Define constants
IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
percent_of_expect=0.56
NUM_FEATURES = 2048
MODEL_PATH = "model/model.keras"
#MODEL_PATH="C:\Main Project 1\LSTM.keras"

# Function definitions
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video_n(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def prepare_single_video_n(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def sequence_prediction(path):
    frames = load_video_n(path)
    frame_features, frame_mask = prepare_single_video_n(frames)
    percent= saved_model.predict([frame_features, frame_mask])[0]
    print(percent)
    return percent

# Load the saved model
MODEL_PATH="C:\Main Project 1\model.keras"
saved_model = keras.models.load_model(MODEL_PATH)
feature_extractor = build_feature_extractor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the path of the uploaded video
        video_file = request.files['video']
        video_path = "uploads/" + video_file.filename
        video_file.save(video_path)

        # Perform sequence prediction
        result = sequence_prediction(video_path)
        

        # Delete the uploaded video after prediction
        os.remove(video_path)

        # Determine the prediction class
        prediction_class = "FAKE" if result <= percent_of_expect else "REAL"
        print(prediction_class)
        

        # Render the result template with the prediction
        return render_template('result.html', prediction=prediction_class)
        #print(prediction_class)

if __name__ == "__main__":
    app.run(debug=True)
