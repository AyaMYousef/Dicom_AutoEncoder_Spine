FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .


RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir tensorflow==2.18.0 \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
