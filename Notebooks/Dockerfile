FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install dependencies:
COPY requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN apt-get install -y libgl1-mesa-glxs