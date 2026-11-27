FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_TF=false
RUN if [ "$INSTALL_TF" = "true" ]; then pip install --no-cache-dir -r requirements-dl.txt; fi
COPY . .
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000 8501
CMD ["uvicorn", "infraguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
