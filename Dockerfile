FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    fastapi==0.141.1 \
    uvicorn==0.54.0 \
    pandas==3.0.6 \
    numpy==2.5.3 \
    scikit-learn==1.9.1 \
    joblib==1.6.0 \
    xgboost==3.4.1

COPY api ./api
COPY models/best_model.joblib ./models/best_model.joblib

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]