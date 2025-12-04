import numpy as np
import pydicom
from PIL import Image
import cv2

def preprocess_dicom(file):
    """
    Read a DICOM file and preprocess for the autoencoder.

    Args:
        file: A file-like object (UploadedFile from FastAPI or open file).

    Returns:
        np.ndarray: Preprocessed image of shape (height, width, 1), dtype float32.
    """
    # Read DICOM
    ds = pydicom.dcmread(file)
    img = ds.pixel_array.astype(np.float32)

    # Normalize to 0-1
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

    # Resize to 256x256 (or your model's input size)
    img = cv2.resize(img, (256, 256))

    # Ensure shape is H x W x 1
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    return img.astype(np.float32)

def preprocess_image(file_bytes):
    img = Image.open(file_bytes).convert("L")  # grayscale
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    return img
