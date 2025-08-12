import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
from PIL import Image, ImageFont, ImageDraw
import pyttsx3
import threading

# ----------------------------
# Threaded webcam capture class
# ----------------------------
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            check, frame = self.stream.read()
            if not check:
                self.stop()
                return
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.stream.release()


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
# Main loop
# ----------------------------
stream = WebcamStream(0).start()
last_pred_time = time.time()
prediction_label = " "

try:
    while True:
        frame = stream.read()
        if frame is None:
            continue

        # Draw ROI rectangle
        cv2.rectangle(frame, (region_left, region_top), (region_right, region_bottom), (0, 0, 0), 2)
        region = frame[region_top:region_bottom, region_left:region_right]

        # Prediction every 5 seconds
        if time.time() - last_pred_time >= 5:
            prediction_label, trans_label = classify_image(region)
            threading.Thread(target=text_to_speech, args=(trans_label,), daemon=True).start()
            last_pred_time = time.time()

        # Render Gujarati text
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((region_left, region_top - 35), f"અનુમાનિત અક્ષર  : {prediction_label} ", font=font, fill=(0, 0, 0))
        frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        cv2.imshow("Capturing", frame_with_text)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting...")

stream.stop()
cv2.destroyAllWindows()
