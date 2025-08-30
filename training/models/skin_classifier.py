import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = r"training\models\mobilenetv2_skin_type_model.keras"
TARGET_SIZE = (224, 224)
FALLBACK_CLASS = 'NOT DETECTED RESCAN YOUR FACE'
THRESHOLD = 0.5
labels = {0: 'Combination', 1: 'Dry', 2: 'Normal', 3: 'Oily', 4: 'Sensitive'}

model = load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def check_lighting(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > 80

def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, TARGET_SIZE)
    face_array = img_to_array(face_img)
    face_array = preprocess_input(face_array)
    return np.expand_dims(face_array, axis=0)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    margin = 0.1
    x1 = max(int(x + w * margin), 0)
    y1 = max(int(y + h * margin), 0)
    x2 = min(int(x + w * (1 - margin)), image.shape[1])
    y2 = min(int(y + h * (1 - margin)), image.shape[0])
    return image[y1:y2, x1:x2]

def predict_skin_type(face_img):
    input_data = preprocess_face(face_img)
    preds = model.predict(input_data, verbose=0)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]
    final_label = labels[top_idx] if confidence >= THRESHOLD else FALLBACK_CLASS
    return final_label, float(confidence), {labels[i]: float(p) for i, p in enumerate(preds)}

