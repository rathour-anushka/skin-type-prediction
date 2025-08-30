import cv2
import numpy as np
from training.models import skin_classifier
from training.util.util import decode_base64_image
from fastapi import UploadFile


def predict_skin_from_upload(file: UploadFile):
    contents = file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


    face_crop = skin_classifier.detect_face(image)
    if face_crop is None:
        return {"error": "No face detected"}

    is_bright = skin_classifier.check_lighting(face_crop)
    preprocessed_crop = skin_classifier.preprocess_face(face_crop)
    label, confidence, probs = skin_classifier.predict_skin_type(preprocessed_crop)

    return {
        "predicted_skin_type": label,
        "confidence": round(confidence, 4),
        "lighting_ok": bool(is_bright),
        "probabilities": probs
    }

def predict_skin_from_webcam():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return {"error": "Webcam not accessible"}

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"error": "Failed to capture image from webcam"}

    face_crop = skin_classifier.detect_face(frame)
    if face_crop is None:
        return {"error": "No face detected"}

    is_bright = skin_classifier.check_lighting(face_crop)
    preprocessed_crop = skin_classifier.preprocess_face(face_crop)
    label, confidence, probs = skin_classifier.predict_skin_type(preprocessed_crop)

    return {
        "predicted_skin_type": label,
        "confidence": round(confidence, 4),
        "lighting_ok": bool(is_bright),
        "probabilities": probs
    }

def predict_skin_from_base64(image_base64: str):
    try:
        image = decode_base64_image(image_base64)
    except ValueError as e:
        return {"error": str(e)}

    face_crop = skin_classifier.detect_face(image)
    if face_crop is None:
        return {"error": "No face detected"}

    is_bright = skin_classifier.check_lighting(face_crop)
    preprocessed_crop = skin_classifier.preprocess_face(face_crop)
    label, confidence, probs = skin_classifier.predict_skin_type(preprocessed_crop)

    return {
        "predicted_skin_type": label,
        "confidence": round(confidence, 4),
        "lighting_ok": bool(is_bright),
        "probabilities": probs
    }