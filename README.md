# Malatec
Faster, better and cheaper mobile malaria detection

![Unet segmentation, yolo detection, mixed models for classification](https://github.com/danielbarco/malatec/blob/main/Notebooks/Images/Malatec.png 'Unet segmentation, yolo detection, mixed models for classification')  

*The image shows the cut sample image taken with a 3D printed openflexure microscope, segmented with a unet, Yolo2 red blood cell detection, and then final classification using a combination of a VGG19 and a SqueezeNet*

To run the Notebooks, first install Docker. 

https://www.tensorflow.org/install/docker?hl=uk \
https://hub.docker.com/r/tensorflow/tensorflow/

You will also need the the image tensorflow/tensorflow:latest-gpu-jupyter

```bash
docker pull tensorflow/tensorflow
docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

Then run the following in the folder were you have the Dockerfile to build the docker file:

```bash
docker build -t container_malatec .
```
Replace PATH with the path to the malatec container and run the following to start up the docker container
```bash
docker run -it --runtime=nvidia --memory="12g" --memory-swap=-1 --oom-kill-disable --rm --name tf_malatec -v /home/fight/Documents/malatec:/tf -p 8888:8888/tcp -p 6006:6006/tcp container_malatec:latest 
```
Now you can open the jupyter notebook (link in the terminal) and navigate to the notebook you would like to work on.

____________________________________________________

YOLO v2 running on Tensorflow 2 from here

https://github.com/jmpap/YOLOV2-Tensorflow-2.0

Malaria Detection Models (ensemble) from NIH

https://github.com/sivaramakrishnan-rajaraman/Deep-Neural-Ensembles-toward-Malaria-Parasite-Detection-in-Thin-Blood-Smear-Images

Unet from

https://www.kaggle.com/advaitsave/tensorflow-2-nuclei-segmentation-unet

____________________________________________________

Cell images from: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets

Microscope images with polygon bounding boxes from: ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/NIH-NLM-ThinBloodSmearsPf/

____________________________________________________

Folder Structure looks something like this

```bash
.
├── ... 
├── Notebooks
├── src
├── data                    # all data
│   ├── cell_images         # images of cut out cells
│   ├── weights             # tensorflow weights
│   ├── stacking            # stacked images
│   ├── pickled             # pickled dataframes and dictionaries
│   ├── masks               # maskes of unet output
│   └── cropped             # cropped and downscaled images 0.3 of original
└── ...
```









