# Use the official Airflow image as the base image
FROM apache/airflow:2.6.0

# Set up environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (e.g., build-essential, wget, curl)
USER root
RUN apt-get update && apt-get install -y \
    wget curl unzip build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Copy requirements.txt to the image and install dependencies
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Set the working directory
WORKDIR /app
COPY . /app
