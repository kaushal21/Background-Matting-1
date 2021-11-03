import argparse
import os
import cv2
import numpy as np


# Tasks for testing:
# 1. Create a separate folder with the data from merged_test in a sub directory input.
#       This stores the input for the testing
# 2. Rename the "_comp" files to "_img" files.
# 3. Create maskDL for all the images using the segmentation
# 4. Call the test_background_matting_image.py to generate the output for the images that we want.
#       We only want the alpha output to calculate the MSE and SAD scores.
#       Thus pass --testing as true to only generate the alpha output
# 5. Call the testing.py on the outputs generated to calculate the MSE and SAD score for the model

# Runnable Command: python testing.py --out_path test/output --mask_path Data_adobe/mask_test
parser = argparse.ArgumentParser(description='compute loss on predicted images from the model')

parser.add_argument('--out_path', type=str, required=True, help='path to generated alpha mattes')
parser.add_argument('--mask_path', type=str, required=True, help='path to provided alpha mattes')
parser.add_argument('--num_bgs', type=int, default=100, help='number of backgrounds onto which to paste each foreground')
args = parser.parse_args()


def loss_MSE(ground_truth, derived_image):
    sub = np.subtract(ground_truth, derived_image)
    square = np.square(sub)
    loss = square.mean()
    return loss


def loss_SAD(ground_truth, derived_image):
    sub = np.subtract(ground_truth, derived_image, dtype=np.float)
    absolute = np.abs(sub)
    loss = 0
    for i in range(ground_truth.shape[2]):
        loss += np.sum(absolute[:][i])
    loss /= ground_truth.shape[1] * ground_truth.shape[0]
    return loss


# Input Data Path
out_path, a_path, num_bgs = args.out_path, args.mask_path, args.num_bgs

out_files = os.listdir(out_path)
a_files = os.listdir(a_path)
loss_mse = {}
loss_sad = {}

for n in range(0, len(out_files), 4):
    filename = out_files[n]

    # Read the generated alpha output
    alpha_out = cv2.imread(os.path.join(out_path, filename.replace('_compose', '_out')))
    alpha_out = cv2.cvtColor(alpha_out, cv2.COLOR_BGR2RGB)

    # Get the name of the image
    gt_temp = filename.split('_')
    gt = ''
    for i in range(0, len(gt_temp)-2):
        gt += gt_temp[i] + '_'

    gt = gt[0: len(gt)-1]

    # Read the ground truth alpha matte for the calculated name for the image
    alpha_gt = cv2.imread(os.path.join(a_path, gt + '.png'))
    alpha_gt = cv2.cvtColor(alpha_gt, cv2.COLOR_BGR2RGB)

    # Calculate the MSE loss
    mse = loss_MSE(ground_truth=alpha_gt, derived_image=alpha_out)
    sad = loss_SAD(ground_truth=alpha_gt, derived_image=alpha_out)
    # Store MSE score
    if gt not in loss_mse:
        loss_mse[gt] = mse
    else:
        loss_mse[gt] += mse

    # Store SAD score
    if gt not in loss_sad:
        loss_sad[gt] = sad
    else:
        loss_sad[gt] += sad

print("Testing Done", len(loss_mse), len(loss_sad))
print("\nMSE Losses: ")
for t in loss_mse:
    print(loss_mse[t])

print("\nSAD Losses: ")

for t in loss_sad:
    print(loss_sad[t])
for t in loss_mse:
    loss_mse[t] /= 100

for t in loss_sad:
    loss_sad[t] /= 100

print("\nAverage MSE: ", sum(loss_mse.values()) / len(loss_mse))
print("Average SAD: ", sum(loss_sad.values()) / len(loss_sad))



