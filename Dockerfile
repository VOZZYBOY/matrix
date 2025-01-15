# Use the official Python image as a base
FROM python:3.9-slim

# Update and install curl
RUN apt-get update && apt-get install -y curl && apt-get clean

#Update and install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Install Yandex Cloud CLI
RUN curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash && \
    mv /root/yandex-cloud/bin/yc /usr/local/bin/yc

# Verify installation
RUN yc --version

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the application port
EXPOSE 8001

# Start the application
CMD ["uvicorn", "matrix:app", "--host", "0.0.0.0", "--port", "8001"]
