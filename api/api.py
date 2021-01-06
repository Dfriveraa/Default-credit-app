from fastapi import APIRouter

from api.endpoints import model

api_router = APIRouter()
api_router.include_router(model.router, prefix="/predict", tags=["predict"])
