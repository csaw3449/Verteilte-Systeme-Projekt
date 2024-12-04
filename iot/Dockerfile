# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY iotDevise.py /app
COPY /data/wisenet_dataset/video_sets /app/data/wisenet_dataset/videosets

# Install dependencies for openCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install OpenCV
RUN pip install opencv-python

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run iotDevise.py when the container launches
ENTRYPOINT ["python", "iotDevise.py"]