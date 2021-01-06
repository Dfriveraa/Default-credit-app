from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from api.api import api_router

app = FastAPI()
origins = ["*"]
app.include_router(api_router, prefix="/api")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
