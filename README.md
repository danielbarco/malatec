# Malatec
Faster, better and cheaper mobile malaria detection

To start the Docker container first install Docker. 

https://www.tensorflow.org/install/docker?hl=uk \
https://hub.docker.com/r/tensorflow/tensorflow/

You will also need the the image tensorflow/tensorflow:latest-gpu-jupyter

```bash
docker pull tensorflow/tensorflow
docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

Then run the following in the folder were you have the Dockerfile:

```bash
docker build -t container_malatec .
```
Replace PATH with the path to the malatec container and run the following
```bash
docker run -it --runtime=nvidia --rm --name tf_malatec -v {PATH}/malatec:/tf -p 8888:8888/tcp -p 6006:6006/tcp container_malatec:latest
```
YOLO v2 running on Tensorflow 2 from here

https://github.com/jmpap/YOLOV2-Tensorflow-2.0






