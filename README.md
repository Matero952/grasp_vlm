<h1 align="center"><strong>Grasp Prediction with Vision-Language Models and Open-Vocabulary Object-Detection Models</strong></h1>

<h2 align="left">Getting Set Up</h2>


```bash
#First, clone the repository
git clone https://github.com/Matero952/grasp_vlm.git
#Then prepare to download the dataset from roboflow, which can be viewed publicly on Roboflow Universe!
cd; cd Downloads
#Then, download the dataset from Roboflow.
curl -L "https://universe.roboflow.com/ds/68uOhAw4FO?key=MDbLOu2FRo" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
#Afterwards, remove unnecessary files.
rm README.dataset.txt; rm README.roboflow.txt
#Then, move 'train' to the directory that the cloned repo is in and rename 'train' to 'grasp_vlm_dataset'
mv train/ replace_this_with_where_you_cloned_the_repo/grasp_vlm/grasp_vlm_dataset
#Done!

<h2 align="left">How to View on Roboflow Universe</h2>




