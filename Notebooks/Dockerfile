FROM tensorflow/tensorflow:latest-gpu-jupyter

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx