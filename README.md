# Gallica BNF app by EPITA 

This respository outlines our end-of-study project on the segmentation of graphic content (texts, images, etc.) for the French National Library (BNF). We have explored and integrated Deep Learning segmentation models such as YoloV8, Detectron2, and LayoutParser. Detailed tutorials are provided to facilitate the use of these models and the metric Cocopanoptic. Furthermore, a visualization application has been developed to enable practical testing of Detectron2 and LayoutParser. This application was created in Python using Gradio, thus offering an intuitive user interface for testing the models on images.


### Authors : 
Vincent THONG, Zoe Sellos, Yanis Farhat, Benjamin Clene, Maxime Boy-Arnould
### Supervised by: 
Jean-Philippe Moreux (BNF),
Joseph Chazalon (EPITA)
## Requirements to run the environment 
- Nvidia GPU
  Nvidia Container Toolkit : https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- 16Go Ram is the minimum to train a dataset (not enough I think)
- Docker Image Size = 15-20 Go
### Docker Tutorial 
- https://docs.docker.com/get-started/
- Docker: https://docs.docker.com/get-docker
  Docker supported GPU : https://docs.docker.com/config/containers/resource_constraints/#gpu

### Command to run:
Building image : docker compose build

Start container :  docker compose up

After this commands a link is displayed with a tocken, it is necessary to go to this link. 
The port is localhost:8888.
You can also access to the environment in VScode by attaching it with Docker Extension.

### Running the gradio app 
cd app; python3 app.py

The port to access is http://localhost:7860/   (a link will appear with 0000:7860 this one doesn't work)

### The app functionnalities : 
Only "Evaluate an image" tab work , only on Detectron2 & LayoutParser
Default classes has 
#TODO : 
- debug gallery
- add metrics cocopanoptic
- add more visualisation
