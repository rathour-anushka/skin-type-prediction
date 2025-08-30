from fastapi import APIRouter, File, UploadFile, HTTPException
from training.services.skin_services import predict_skin_from_upload, predict_skin_from_webcam
from training.services.skin_services import predict_skin_from_base64
from training.schema.skin_schema import Base64ImageRequest

router = APIRouter()

@router.post("/skin/predict")
async def predict_skin_type(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Unsupported file format.")
    result = predict_skin_from_upload(file)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.get("/skin/predict/webcam")
def predict_skin_type_from_webcam():
    result = predict_skin_from_webcam()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.post("/skin/predict/webcam/base64")
async def predict_skin_type_from_webcam_base64(payload: Base64ImageRequest):
    result = predict_skin_from_base64(payload.image_base64)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result