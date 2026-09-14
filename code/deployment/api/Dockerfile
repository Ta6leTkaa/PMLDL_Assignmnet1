FROM python:3.14-slim
WORKDIR /workspace
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY code/deployment/api/main.py ./main.py
COPY models/model.joblib ./models/model.joblib
ENV MODEL_PATH=/workspace/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
