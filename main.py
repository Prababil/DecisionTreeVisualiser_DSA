
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
import pandas as pd
from starlette.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from flask import Flask,redirect
from process import get_decision_tree
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name='images')

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/fileUpload")
async def create_upload_file(dataset: UploadFile = File(...)):
    if dataset.filename.endswith(".csv"):
        df = pd.read_csv(dataset.file)
        summary_stats = df.describe()
        get_decision_tree(df)
        return FileResponse("response.html")
    else:
        return