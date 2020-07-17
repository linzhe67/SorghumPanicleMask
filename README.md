# Sorghum Panicle Detection and Counting using Unmanned Aerial System Images and Deep Learning

This is a support page of paper "Sorghum Panicle Detection and Counting using Unmanned Aerial System Images and Deep Learning" as an original research. We developed a sorghum panicle detection and counting pipeline using UAS images based on an integration of image segmentation and UNet CNN model.

## `Code`
UNet algorithm training code was based on Ronneberger et al., 2015 research and modified to fit our objectives.

- code for training models of the UNet in Python
  - `Model_training.py`
  
- code for counting of the UNet in Python
  - `Panicle_count.py`
  
- code for spliting testing images in Python
  - `image_spliting.py`
  
## `Training_dataset`
- Sorghum training images (1024 x 1024)
  - `images`
  
- Sorghum training images corresponding masks (1024 x 1024)
  - `masks`
  
## `Testing_dataset`
Before testing, please split all test images.

- Sorghum testing images
  - `testing`
  
### `To download the full dataset, please send me a email about your name and institution. I will send you the download link`
