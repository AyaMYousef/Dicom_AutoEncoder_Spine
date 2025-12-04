from http.client import HTTPException
from app.model import predict_anomaly
from fastapi import FastAPI, UploadFile, File
from app.utils import preprocess_dicom, preprocess_image


app = FastAPI(title="DICOM Autoencoder Anomaly Detection")

@app.get("/healtycheck")
def read_root():
    return {"Hello": "World"}

@app.get("/")
def root():
    return {"message": "Welcome to the DICOM Autoencoder API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.01):
    """Upload a DICOM or image file and return anomaly prediction."""

    filename = file.filename.lower()

    # Detect file type
    if filename.endswith(".dcm"):
        img = preprocess_dicom(file.file)

    elif filename.endswith((".png", ".jpg", ".jpeg")):
        img = preprocess_image(await file.read())

    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload .dcm, .png, .jpg, .jpeg"
        )

    # Run prediction
    result = predict_anomaly(img, threshold)
    return result