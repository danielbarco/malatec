# following https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
FROM tensorflow/tensorflow:latest-gpu-jupyter
WORKDIR /Notebooks/
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3-venv -y
RUN python3 -m venv /opt/venv

# Install dependencies:
COPY /Notebooks/requirements_tf.txt .
RUN . /opt/venv/bin/activate 
RUN pip install --upgrade pip 
RUN pip install -r requirements_tf.txt

EXPOSE 8888 6006