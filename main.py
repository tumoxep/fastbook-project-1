from fastai import learner
from fastai.vision.core import PILImage
from fastapi import FastAPI, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
learn_inf = learner.load_learner('export.pkl')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    img = PILImage.create(file.file)
    pred,pred_idx,probs = learn_inf.predict(img)
    return {"data": f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'}

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
