import os
import glob
import argparse

import torch
import torch.nn as nn
from skimage.measure import label
from torch.autograd import Variable

from data_loader import create_segmentation_guide
from functions import *
from functions import get_bounding_box
from networks import Network

# Tasks for Generating the output:
# 1. Read all the directory paths from the user for input data, background data and the output location
# 2. Initialize the network with the model pass
# 3. Read the images from the input data and perform for each input image
#     3.1 Load the Image, its background, segmentation mask, target background
#	  3.2 Get Bounding box from Segmentation Mask and crop all the images according the this Bounding Box
#	  3.3 Create the Segmentation Guide from this cropped segmentation mask
#	  3.4 Convert all image from numpy to torch
#	  3.5 Pass the Image through the Network
# 	  3.6 Extract image where value of alpha is higher
# 	  3.7 Smoothen out the alpha
# 	  3.8 Convert the images from numpy to images.
#	  3.9 Un-crop the images / resize them to the original size
#	  3.10 Save the all the output images

# Runnable Command: python generate_output_images.py -m train-adobe-40 -i data/input -o data/output -tb data/bg/001.png
parser = argparse.ArgumentParser(
    description='Runs the model on the given inputs to generate new output with background matting.')

parser.add_argument('-m', '--trained_model', type=str, default='train-adobe-64',
                    help='Trained background matting model')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='Directory to save the output results. (required)')
parser.add_argument('-i', '--input_dir', type=str, required=True, help='Directory to load input images. (required)')
parser.add_argument('-tb', '--target_back', type=str, help='Directory to load the target background.')
args = parser.parse_args()

# Input Data Path
data_path = args.input_dir
# Output Data Path
result_path = args.output_dir
# Input Model Path
model_dir = 'Models/' + args.trained_model + '/'

############################ Loading Data ############################
# Get all the unique images from the input directory.
input_img = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and i.endswith('_img.png')]
input_img.sort()

# Loading Background that would be applied
bg_img = cv2.imread(args.target_back); bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
# New blank green-screen background with size similar to target background
green_bg = np.zeros(bg_img.shape)

# Creating Output Directory
if not os.path.exists(result_path):
	os.makedirs(result_path)

############################ Initializing the Network ############################

try:
	net = glob.glob(model_dir + 'net_epoch_*')
	model = net[0]
	network = Network(input=(3, 3, 1, 4), output=4, n_block1=5, n_block2=2)
	network = nn.DataParallel(network)
	network.load_state_dict(torch.load(model))
	network.cuda()
	torch.backends.cudnn.benchmark = True
	network.eval()
	resolution = (256, 256)  # input resolution to the network
except Exception as exception:
	print('Error While Loading: ', model_dir)
	print(exception)

############################ Running Network on each Image ############################
for i in range(0, len(input_img)):
	filename = input_img[i]
	# Original Image
	image_or = cv2.imread(os.path.join(data_path, filename)); image_or = cv2.cvtColor(image_or, cv2.COLOR_BGR2RGB)
	# Original Background Image
	bg_or = cv2.imread(os.path.join(data_path, filename.replace('_img', '_back'))); bg_or = cv2.cvtColor(bg_or, cv2.COLOR_BGR2RGB)
	# Generated Segmentation Mask
	seg_or = cv2.imread(os.path.join(data_path, filename.replace('_img', '_masksDL')), 0)

	# Get the bounding box from segmentation and Crop all the images according to the bounding box
	image_updated = image_or
	bbox = get_bounding_box(seg_or, A=image_or.shape[0], B=image_or.shape[1])

	image_updated = crop_image(image_updated, resolution, bbox)
	bg_im = crop_image(bg_or, resolution, bbox)
	seg_or = crop_image(seg_or, resolution, bbox)
	back_img1 = crop_image(bg_img, resolution, bbox)
	back_img2 = crop_image(green_bg, resolution, bbox)

	# Process Segmentation Mask
	seg_or = create_segmentation_guide(seg_or, resolution, ksize=True)

	# Convert Images to Torch to run them in the Network
	image_torch = torch.from_numpy(image_updated.transpose((2, 0, 1))).unsqueeze(0); image_torch = 2 * image_torch.float().div(255) - 1
	bg_torch = torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0); bg_torch = 2 * bg_torch.float().div(255) - 1
	seg_torch = torch.from_numpy(seg_or).unsqueeze(0).unsqueeze(0); seg_torch = 2 * seg_torch.float().div(255) - 1

	# Run the image through the network
	with torch.no_grad():
		image_torch, bg_torch, seg_torch = Variable(image_torch.cuda()), Variable(bg_torch.cuda()), Variable(seg_torch.cuda())

		alpha_pred, fg_pred = network(image_torch, bg_torch, seg_torch)

		# Keep the image as foreground where alpha is greater
		alpha_mask = (alpha_pred > 0.95).type(torch.cuda.FloatTensor)
		fg_pred = image_torch * alpha_mask + fg_pred * (1 - alpha_mask)

		alpha_out = convert_to_image(alpha_pred[0, ...])

		# Label the connected layout of alpha as one
		labels = label((alpha_out > 0.05).astype(int))
		try:
			assert (labels.max() != 0)
		except AssertionError as exception:
			continue
		# Get the largest label and multiply alpha out to get a smoother alpha output
		largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
		alpha_out = alpha_out * largestCC

		alpha_out = (255 * alpha_out[..., 0]).astype(np.uint8)

		# Convert foreground image from numpy to image
		fg_out = convert_to_image(fg_pred[0, ...])
		fg_out = fg_out * np.expand_dims((alpha_out.astype(float) / 255 > 0.01).astype(float), axis=2)
		fg_out = (255 * fg_out).astype(np.uint8)

		# Un-Crop the outputs into the original size
		R = image_or.shape[0]; C = image_or.shape[1]
		alpha_out = un_crop(alpha_out, bbox, R, C)
		fg_out = un_crop(fg_out, bbox, R, C)

	# Resize the images to the original sizes
	bg_img = cv2.resize(bg_img, (C, R))

	# Compose Image using the equation for fg, bg and alpha
	composite_image = composing_image(fg_out, bg_img, alpha_out)

	# Save the images i.e, composite image, foreground, segmentation mask, alpha matte
	cv2.imwrite(result_path + '/' + filename.replace('_img', '_out'), alpha_out)
	cv2.imwrite(result_path + '/' + filename.replace('_img', '_fg'), cv2.cvtColor(fg_out, cv2.COLOR_BGR2RGB))
	cv2.imwrite(result_path + '/' + filename.replace('_img', '_compose'), cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB))

	print('Done: ' + str(i + 1) + '/' + str(len(input_img)))
