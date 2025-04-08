# Use the official Python image as a base
FROM python:3.9-slim

# Update and install curl and ffmpeg
# Update and install curl, ffmpeg, build-essential AND git
# Update and install curl, ffmpeg, build-essential AND git

RUN apt-get update && \
    apt-get install -y curl ffmpeg build-essential git && \
    apt-get clean
    

# Install Yandex Cloud CLI
RUN curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash && \
    mv /root/yandex-cloud/bin/yc /usr/local/bin/yc

# Verify installation
RUN yc --version

# Set working directory
WORKDIR /app

# Copy just the requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --quiet flit && \
    pip install --no-cache-dir --quiet -I git+https://github.com/yandex-cloud/yandex-cloud-ml-sdk.git@assistants_fc#egg=yandex-cloud-ml-sdk && \
    pip install --no-cache-dir --upgrade --quiet pydantic

# Make sure the base directory exists
RUN mkdir -p /app/base

# Copy the application files
COPY app.py matrixai.py ./
COPY base/cleaned_data.json ./base/

# Expose the application port
EXPOSE 8001

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
