# Dockerfile builds the FastAPI backend only.
# (The Streamlit frontend is deployed separately on Streamlit Community Cloud —
#  see README for why that split is the easiest free deployment path.)

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
