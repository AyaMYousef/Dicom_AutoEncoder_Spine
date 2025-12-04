# DICOM AutoEncoder - Spine Anomaly Detection

## Description

This project uses a convolutional autoencoder to identify abnormal patterns in spinal DICOM images. The model reconstructs input images and computes a reconstruction error; images with high error are flagged as anomalies.

It provides a **FastAPI** endpoint for inference and is fully containerized with **Docker**, enabling easy deployment.

---

## Features

- Train and save AutoEncoder for spinal DICOM images
- Detect anomalies using reconstruction error
- FastAPI API for inference
- Dockerized for seamless deployment
- CI/CD integration via GitHub Actions

---

## Installation

### Clone repository

```bash
git clone https://github.com/AyaMYousef/Dicom_AutoEncoder_Spine.git
cd Dicom_AutoEncoder_Spine
```

### Using Docker

1. Build Docker image:

```bash
docker build -t dicom-autoencoder:latest .
```

2. Run the container:

```bash
docker run -p 8000:8000 dicom-autoencoder:latest
```

The FastAPI server will be available at `http://localhost:8000`.

---

## API

**POST** `/predict?threshold=<value>`

- **Request**: send a DICOM image as input
- **Response**: JSON containing:

  - `reconstruction_mse`: float
  - `is_anomaly`: boolean

Example:

```json
{
  "reconstruction_mse": 0.023,
  "is_anomaly": true
}
```

---

## GitHub Actions

- The project includes a CI/CD workflow to:

  1. Build Docker image
  2. Run tests (if configured)
  3. Publish image to GitHub Container Registry (GHCR)

---

## Requirements

- Python 3.11
- TensorFlow 2.18.0
- FastAPI, Uvicorn
- OpenCV, Pydicom, NumPy, Pillow
- Docker

---

## Docker Image

- Built and published via GitHub Actions
- Pull image:

```bash
docker pull ghcr.io/<your-username>/dicom-autoencoder:latest
```

---
