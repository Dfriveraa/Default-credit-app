from fastapi import APIRouter
from fastapi import File, UploadFile
import pandas as pd
from Classifier import Classifier
import shutil

router = APIRouter()
clf = Classifier()

@router.post("/files/")
async def create_file(file: bytes = File(...)):
    print(file.decode())
    return {"file_size": len(file)}


@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    a=pd.read_csv(file.file)

    # with open(file.filename, "wb") as buffer:
    #     shutil.copyfileobj(file.file, buffer)
    #     a=pd.read_csv(file.filename)
    #     print(a.head(3))
    result=clf.predict(a)
    return result



