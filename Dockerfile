FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless==4.13.0.92

COPY . .

CMD ["python", "batch_test.py"]