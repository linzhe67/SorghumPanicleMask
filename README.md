# Sorghum Panicle Detection and Counting using Unmanned Aerial System Images and Deep Learning

This is a support page of paper "Sorghum Panicle Detection and Counting using Unmanned Aerial System Images and Deep Learning" as an original research. We developed a sorghum panicle detection and counting pipeline using UAS images based on an integration of image segmentation and UNet CNN model.

## `Code`
UNet algorithm was based on Ronneberger et al., 2015 research and modified to fit our objectives.

- code for training models of the UNet in Python
  - `Model_training.py`
  
- code for counting of the UNet in Python
  - `Panicle_count.py`
  
- code for spliting testing images in Python
  - `image_spliting.py`

- Steps:
  - `Run Model_training.py with Training_dataset`
  - `Split all testing images using image_spliting.py`
  - `Run Panicle_count.py with splited testing images`
  - `Please check the detailed information in the manuscript`


## `Training_dataset`
- Sorghum training images (1024 x 1024)
  - `images`
  
- Sorghum training images corresponding masks (1024 x 1024)
  - `masks`
  
## `Testing_dataset`
- Sorghum testing images
  - `testing`

## `Full training and testing dataset`
Due to the Github size limit, we cannot upload the full dataset. To download the full dataset, please use the link below: 

