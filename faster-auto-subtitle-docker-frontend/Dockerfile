# -------- Stage 1: Builder --------
FROM python:3.12-alpine AS builder

# Prevents interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build tools only for compiling dependencies
RUN apk add --no-cache gcc musl-dev libffi-dev

# Install only required Python packages
COPY app.py requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt


# -------- Stage 2: Minimal Runtime --------
FROM python:3.12-alpine

WORKDIR /app

# Add runtime dependency for requests (SSL, etc.)
RUN apk add --no-cache libffi openssl

# Copy installed Python libs from builder
COPY --from=builder /install /usr/local
COPY . .

CMD ["python", "app.py"]

