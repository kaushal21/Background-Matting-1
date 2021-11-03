# Background Matting: The World is Your Green Screen

We ran the final version of this code on Google Colab.
In order to do that, we created a GitHub repository where we maintained the code.
This allowed us to track the progress and helped in reverting to a previous version as well.

## Colab Notebook

The Colab Notebook is split into five major parts:

1. **Required setup**: This is the initial part where we clone the git repo and install the requirements.

2. **Data Preparation**: In this part we download the data sets. We make a copy of Adobe Matting Data set for foreground images from google drive and download MS-COCO dataset for background images. Then we compose these two datasets and merge them to have our training set and testing set ready

3. **Training**: We train our model on the default setting from out git repo which we have set according to our observations and also considering the limitations of our running setup. We can change these setting if we want to while running this file.

4. **Testing**: We test our model generated on the merged testing set that we generated in data preparation step. We calculate the MSE and SAD score for each model that we generate

5. **Output step / Final Run**: We run this model to give us real life examples. We pass in input images of foreground, background of the image and the target background. the model generates the final image of foreground copied onto the target background image.

## GitHub repository

The repo consist of 7 files that we created for this project. We have taken the initial segmentation with deeplab from the original authors.
We have done that to have a same starting point as the authors. We have explained this in details within that file itself.

Functionality of each file can be summarized as:

1. data_loader.py: This file helps in loading the data that we have generated using the datasets. It takes in the csv file which is generated while data preparation. It also has some useful functions that comes in handy while performing tasks
2. functions.py: Consider this file as a helper file. It holds several functions that helps in performing tasks like cropping images, getting bounding box, etc
3. loss_function.py: This holds in all the loss that we calculate for this project. Losses like Alpha loss, compose loss and GAN loss
4. network.py: The complete network architecture is within this file. Network, ResBLK are within this file
5. generate_output_images.py: We pass on input images with target background to this file which used a pretrained model to generate a final output. It produces 4 output images, which are the foreground, alpha, a green background alpha matte and a final composed image.
6. train.py: This is the training file, it calls in the data_loader to load the data, trains the network over it and saves a model using losses from loss_functions
7. testing.py: For testing we used the merged dataset that we generated using Adobe Matting Dataset and MS-COCO. We calculate MSE and SAD score for a given model

There are two unchanged files that we have used from authors original code. 

1. segmentation_deeplab.py: This file is used for the segmentation of the input image using pretrained deeplab models
2. compose.py: This file is used for the data generation process. We have requested the authorization form Adobe to use the Adobe Matting dataset for this project. This file masks the images from the adobe matting dataset onto images from MS-COCO as background to create merged training and testing dataset. This is done to have a big enough dataset on which we can train the model. 


