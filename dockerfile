# Use Python 3.10 as base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install system dependencies (Graphviz and pkg-config)
RUN apt-get update && apt-get install -y \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the application files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (if needed)
EXPOSE 8000

# Command to run the application
CMD ["python", "api.py"]
