FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /Notebooks/
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-venv -y
RUN python3 -m venv /opt/venv

# Install dependencies:
COPY /Notebooks/requirements.txt .
RUN . /opt/venv/bin/activate && pip install --upgrade pip 