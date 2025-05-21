# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.12.9
FROM python:${PYTHON_VERSION}-slim AS base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dev into container
COPY src/dev ./app/dev

# Copy the source code into the container.
COPY src/roalt_api/app ./app

# Copy pyproject toml
COPY pyproject.toml .

# Open-CV requirements
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry lock
RUN poetry install --with gpu --with api --no-root --no-interaction --no-ansi

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN mkdir -p logs && chown -R ${UID}:${UID} logs

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8080

# Run the application.
CMD ["python3", "./app/main.py"]

#Root-Analysis-Toolkit>docker build -f ./src/roalt_api/Dockerfile -t roalt_api:latest .
#docker run -p 8000:8000 --name roalt_api_container roalt_api:latest