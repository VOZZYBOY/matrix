# Use the official Python image as a base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install Yandex Cloud CLI
RUN curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash && \
    mv /root/yandex-cloud/bin/yc /usr/local/bin/yc

# Verify installation
RUN yc --version

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the application port
EXPOSE 8001

# Start the application
CMD ["uvicorn", "matrix:app", "--host", "0.0.0.0", "--port", "8001"]
