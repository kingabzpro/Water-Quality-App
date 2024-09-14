from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import skops.io as sio
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Load the saved model and preprocessing pipeline
model = sio.load("models/water_quality_model.skops")
preprocessing_pipeline = joblib.load("models/scaler.joblib")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ph: float = Form(...),
    Hardness: float = Form(...),
    Solids: float = Form(...),
    Chloramines: float = Form(...),
    Sulfate: float = Form(...),
    Conductivity: float = Form(...),
    Organic_carbon: float = Form(...),
    Trihalomethanes: float = Form(...),
    Turbidity: float = Form(...),
):
    input_data = np.array(
        [
            [
                ph,
                Hardness,
                Solids,
                Chloramines,
                Sulfate,
                Conductivity,
                Organic_carbon,
                Trihalomethanes,
                Turbidity,
            ]
        ]
    )
    # Preprocess input data
    input_preprocessed = preprocessing_pipeline.transform(input_data)
    prediction = model.predict(input_preprocessed)
    result = "Potable" if prediction[0] == 1 else "Not Potable"
    return templates.TemplateResponse(
        "result.html", {"request": request, "result": result}
    )
