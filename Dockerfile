# Use the official Python image as a base
FROM python:3.12-slim



# Update and install system dependencies (curl, ffmpeg for potential media processing, build-essential for packages needing compilation, git for git installs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ffmpeg build-essential git ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Yandex Cloud CLI (Keep only if actively used by the application)
RUN curl -sSL https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash && \
    mv /root/yandex-cloud/bin/yc /usr/local/bin/yc

# Verify YC CLI installation (Optional, can be removed)
RUN yc --version

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Upgrade pip, install flit (if needed for yandex sdk?) and then install from requirements.txt
# Clean up the pip install commands
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --quiet flit && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install yandex-cloud-ml-sdk from git if needed (ensure the branch/tag is correct)
    # Upgrade pydantic again separately if a specific version is needed after other installs
    pip install --no-cache-dir --upgrade --quiet pydantic

# Make sure the base directory exists (if cleaned_data.json needs it)

# Copy the application code and necessary files
COPY app.py client_data_service.py matrixai.py clinic_functions.py rag_setup.py redis_history.py tenant_config_manager.py clinic_index.py service_disambiguation.py message_completeness_analyzer.py language_detector.py  ./
COPY base/ ./base/
COPY static ./static/
COPY templates ./templates/
# Expose the application port
EXPOSE 8001

# Start the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
