FROM tensorflow/tensorflow:latest-gpu-jupyter
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/devel-gpu-jupyter.Dockerfile

# Install dependencies:
COPY requirements.txt .
RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/daviddexter/keras_squeezenet2
RUN pip install -q keras_squeezenet2/
# It seems the keras_squeezenet2 installs tf 2.0.0 wheras we want the newest version:
RUN pip install tensorflow -U
RUN pip install tensorboard -U

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    protobuf-compiler \
    wget
EXPOSE 6006
EXPOSE 8888

#Object detection package tf
RUN git clone --depth 1 https://github.com/tensorflow/models
#RUN protoc models/research/object_detection/protos/*.proto --python_out=.
COPY models/research/object_detection/packages/tf2/setup.py .
# code fails if we do not install the following avro-python3
RUN pip install avro-python3==1.9.2.1
RUN python setup.py install

#download and extract tf object detection model to new folder download
WORKDIR /download
RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz 
RUN tar -xf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz 

# RUN tensorboard --logdir logs
# RUN apt-get install -y libgl1-mesa-glxs
# RUN git clone https://github.com/daviddexter/keras_squeezenet2
# RUN pip install -q keras_squeezenet2/
