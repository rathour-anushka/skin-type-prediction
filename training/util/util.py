import base64
import numpy as np
import cv2

def decode_base64_image(image_base64: str):
    try:
        image_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")
