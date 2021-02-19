FROM tensorflow/tensorflow:latest-gpu-jupyter
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-gpu-jupyter.Dockerfile

# Install dependencies:
COPY requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt

RUN git clone https://github.com/daviddexter/keras_squeezenet2
RUN pip install -q keras_squeezenet2/
# It seems the keras_squeezenet2 installs tf 2.0.0 wheras we want the newest version:
RUN pip install tensorflow -U
RUN pip install tensorboard -U

RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
EXPOSE 6006
EXPOSE 8888
# RUN tensorboard --logdir logs
# RUN apt-get install -y libgl1-mesa-glxs
# RUN git clone https://github.com/daviddexter/keras_squeezenet2
# RUN pip install -q keras_squeezenet2/
