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

#Object detection package tf
RUN git clone --depth 1 https://github.com/tensorflow/models
# intall protoc
RUN apt install -y protobuf-compiler
WORKDIR /models/research/
# RUN protoc --version
RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN python -m pip install .
# RUN tensorboard --logdir logs
# RUN apt-get install -y libgl1-mesa-glxs
# RUN git clone https://github.com/daviddexter/keras_squeezenet2
# RUN pip install -q keras_squeezenet2/
