# app/api/router.py
from fastapi import APIRouter
from training.controllers import skin_controller

api_router = APIRouter()
api_router.include_router(skin_controller.router, prefix="/training/skin", tags=["Skin Detection"])
