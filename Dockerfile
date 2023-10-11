# Use the official Python 3.8 image from Docker Hub
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the contents from app folder into the container at /app
ADD app /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make sure scripts in .local are usable:
ENV PATH=/root/.local:$PATH

# Expose port 3111 for the application
EXPOSE 3111

# Define the command to run the app using CMD which defaults to /bin/bash
CMD ["python", "main.py"]
