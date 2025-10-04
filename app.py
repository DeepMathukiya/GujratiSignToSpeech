import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response, redirect, url_for
from PIL import Image, ImageFont, ImageDraw
import time
import threading
import pyttsx3

app = Flask(__name__)

# ----------------------------
# Load model & constants
# ----------------------------
region_top, region_bottom, region_left, region_right = 40, 314, 10, 314
model2 = load_model("newdata.h5")

class_labels = [
    'ક','ખ', 'ગ','ઘ', 'ચ', 'છ', 'જ', 'ટ', 'ઠ', 'ડ', 'ઢ', 'ણ', 'ત', 'થ', 'દ', 'ધ', 'ન',
    'ફ', 'બ', 'ભ', 'મ', 'ય', 'ર', 'લ', 'વ','શ','ષ' , 'સ' ,'હ','ળ','ક્ષ', 'જ્ઞ','0'
]
transliterated_labels = [
    'ka', 'kha', 'ga', 'gha', 'cha', 'chha', 'ja', 'ta', 'tha', 'da', 'dha', 'ana', 'ta', 'tha', 'da', 'dha', 'na',
    'pha', 'ba', 'bha', 'ma', 'ya', 'ra', 'la', 'va', 'sha', 'sh', 'sa', 'ha', 'ala', 'khsa', 'gna', '0'
]

font_path = "NotoSerifGujarati-Regular.ttf"
font = ImageFont.truetype(font_path, 32)

# ----------------------------
# Global variables
# ----------------------------
camera = None
camera_running = False
last_pred_time = time.time()
prediction_label = " "

# ----------------------------
# Helper functions
# ----------------------------
def preprocess_image(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 11, 7, 21)
    img = cv2.resize(img, (224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(img):
    img_array = preprocess_image(img)
    prediction = model2.predict(img_array)
    predicted_class = np.argmax(prediction)
    return class_labels[predicted_class], transliterated_labels[predicted_class]

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ----------------------------
# Frame Generator
# ----------------------------
def generate_frames():
    global last_pred_time, prediction_label, camera, camera_running

    while camera_running and camera is not None:
        success, frame = camera.read()
        if not success:
            break

        # Draw ROI
        cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (0, 0, 0), 2)

        # Predict every 5 seconds
        if time.time() - last_pred_time >= 5:
            region = frame[region_top:region_bottom, region_left:region_right]
            predicted, trans = classify_image(region)
            prediction_label = predicted
            threading.Thread(target=text_to_speech, args=(trans,), daemon=True).start()
            last_pred_time = time.time()

        # Draw Gujarati text
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((region_left, region_top - 35), f"અનુમાનિત અક્ષર : {prediction_label}", font=font, fill=(0, 0, 0))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html', prediction=prediction_label)

@app.route('/video')
def video():
    if camera_running and camera is not None:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera is not running. Please click Start."

@app.route('/start')
def start_camera():
    global camera, camera_running
    if not camera_running:
        camera = cv2.VideoCapture(0)
        camera_running = True
    return redirect(url_for('index'))

@app.route('/stop')
def stop_camera():
    global camera, camera_running
    if camera_running and camera is not None:
        camera_running = False
        camera.release()
        camera = None
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
