# Malatec
Faster, better and cheaper mobile malaria detection

![Move microscope, cellpose segmentation and ensemble model for classification using SqueezeNet + VGG19 ](Notebooks/Images/malatec_mvp2.gif 'Move microscope, cellpose segmentation and ensemble model for classification using SqueezeNet + VGG19')

**[Checkout our app here 🎉 🔬](https://github.com/danielbarco/malatec_app)**

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
docker run -it --runtime=nvidia --memory="16g" --memory-swap=-1 --oom-kill-disable --rm --name tf_malatec -v PATH/malatec:/tf -p 8888:8888/tcp -p 6006:6006/tcp container_malatec:latest 
```
Now you can open the jupyter notebook (link in the terminal) and navigate to the notebook you would like to work on.

____________________________________________________
**Pipeline**    

- Cell segmentation from microscope images using [Article](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1) | [GitHub](https://github.com/MouseLand/cellpose)  
- Classification: SqueezeNet + VGG19 [Article](https://peerj.com/articles/6977.pdf) | [GitHub](https://github.com/sivaramakrishnan-rajaraman/Deep-Neural-Ensembles-toward-Malaria-Parasite-Detection-in-Thin-Blood-Smear-Images)
____________________________________________________
**Data**

- Cell images from: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets

- Microscope images with polygon bounding boxes from: ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/NIH-NLM-ThinBloodSmearsPf/

**[Publicly Available Malria Datasets](https://github.com/danielbarco/malaria_datasets)**

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
Made with ❤️ in Switzerland ⛰️









