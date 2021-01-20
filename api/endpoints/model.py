from fastapi import APIRouter
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
from Classifier import Classifier

some_file_path = "Ejemplo.csv"

router = APIRouter()
clf = Classifier()


@router.get("/example/")
async def get_example():
    return FileResponse(some_file_path)


@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    a = pd.read_csv(file.file, sep=";")
    result = clf.predict(a)
    return result
