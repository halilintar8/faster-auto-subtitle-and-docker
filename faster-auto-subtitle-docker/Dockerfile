FROM my-gpu-base:latest

ENV TRANSFORMERS_CACHE=/cache/huggingface \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy app code (no requirements.txt or pip install needed)
COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
