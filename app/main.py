from fastapi import FastAPI
from training.controllers import skin_controller

app = FastAPI(title="Skin Type Detection API")

app.include_router(skin_controller.router)
