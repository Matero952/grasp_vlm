<h1 align="center"><strong>Grasp Prediction with Vision-Language Models and Open-Vocabulary Object-Detection Models</strong></h1>

## Getting Set Up

```bash
#First, clone the repository and make sure that you have Conda.
git clone https://github.com/Matero952/grasp_vlm.git
#Then prepare to download the dataset from roboflow, which can be viewed publicly on Roboflow Universe!
cd; cd Downloads
#Then, download the dataset from Roboflow.
curl -L "https://universe.roboflow.com/ds/68uOhAw4FO?key=MDbLOu2FRo" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
#Afterwards, remove unnecessary files.
rm README.dataset.txt; rm README.roboflow.txt
#Then, move train to the right directory. Replace the '*' with the cloned repo's path.
mv train/ */grasp_vlm_dataset
#Replace the '*' with the cloned repo's path.
cd *

```

## How to View on Roboflow Universe:
To view the dataset on Roboflow Universe, type the following in a search bar:
```bash
https://universe.roboflow.com/correllmateo/grasp_vlm
```
Here, you can view all images with their cooresponding bounding boxes.




