import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
import cv2


class VideoData(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class AdobeData(Dataset):
    def __init__(self, file, config):
        self.frames = pd.read_csv(file, sep=';')
        self.resolution = config['resolution']
        self.trimap = config['trimapK']
        self.noise = config['noise']

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        try:
            # Read the Image from file, Convert it into RGB and Resize the image
            fg = cv2.imread(self.frames.iloc[item, 0]); fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB); fg = cv2.resize(fg, dsize=(800, 800))
            alpha = cv2.imread(self.frames.iloc[item, 1]); alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2RGB); alpha = cv2.resize(alpha, dsize=(800, 800))
            image = cv2.imread(self.frames.iloc[item, 2]); image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); image = cv2.resize(image, dsize=(800, 800))
            bg = cv2.imread(self.frames.iloc[item, 3]); bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB); bg = cv2.resize(bg, dsize=(800, 800))

            # Randomly flip the image for making the dataset more random. If flipping then flip all images, otherwise skip it
            if np.random.random_sample() > 0.5:
                fg = cv2.flip(fg, 1)
                alpha = cv2.flip(alpha, 1)
                image = cv2.flip(image, 1)
                bg = cv2.flip(bg, 1)

            trimap = generate_trimap(alpha, self.trimap[0], self.trimap[1], False)

            # To generate even more randomness, crop and scale the image randomly from the given sizes,
            # These sizes are kept similar to the original paper to make the images training and testing comparable
            size_options = [(576, 576), (608, 608), (640, 640), (672, 672), (704, 704), (736, 736), (768, 768), (800, 800)]
            randomly_selected_size = random.choice(size_options)

            x, y = random_choice(trimap, randomly_selected_size)

            fg = safe_crop(fg, x, y, randomly_selected_size, self.resolution)
            alpha = safe_crop(alpha, x, y, randomly_selected_size, self.resolution)
            image = safe_crop(image, x, y, randomly_selected_size, self.resolution)
            bg = safe_crop(bg, x, y, randomly_selected_size, self.resolution)
            trimap = safe_crop(trimap, x, y, randomly_selected_size, self.resolution)

            # Add Noise to the background image if passed parameter.
            # Randomly select between noise or gamma change between them.
            if self.noise:
                # Add Noise
                if np.random.random_sample() > 0.5:
                    sigma = np.random.randint(low=2, high=6)
                    mean = np.random.randint(low=0, high=14) - 7
                    back = add_noise(bg, mean, sigma)
                # Adjust Gamma
                else:
                    invGamma = 1.0 / np.random.normal(1, 0.12)
                    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

                    back = cv2.LUT(bg, table)

            # Creating motion queues for the image. Create 4 transformed images using Affine Transformation
            image_affine = np.zeros((fg.shape[0], fg.shape[1], 4))
            for i in range(0, 4):
                theta = np.random.normal(0, 7)
                rotation = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
                                     [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
                scale = np.array([[1 + np.random.normal(0, 0.05), 0], [0, 1]])
                shear = np.array([[1, np.random.normal(0, 0.05) * (np.random.random_sample() > 0.5)],
                                 [np.random.normal(0, 0.05) * (np.random.random_sample() > 0.5), 1]])
                adjustment = np.concatenate((scale * shear * rotation, np.random.normal(0, 5, (2, 1))), axis=1)

                # New foreground with adjustment
                new_fgi = cv2.warpAffine(fg.astype(np.uint8), adjustment, (fg.shape[1], fg.shape[0]), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)
                # New Alpha with adjustment
                new_alphai = cv2.warpAffine(alpha.astype(np.uint8), adjustment, (fg.shape[1], fg.shape[0]),
                                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

                sigma = np.random.randint(low=2, high=6)
                mean = np.random.randint(low=0, high=14) - 7
                new_bg1 = add_noise(bg, mean, sigma)

                image_affine[..., i] = cv2.cvtColor(composite(new_fgi, new_bg1, new_alphai), cv2.COLOR_BGR2GRAY)

            # Creating final dictionary with the images
            sample = {'fg': to_tensor(fg),
                      'alpha': to_tensor(alpha),
                      'bg': to_tensor(bg),
                      'image': to_tensor(image),
                      'segmentation': to_tensor(create_segmentation(alpha, trimap)),
                      'trimap': to_tensor(trimap),
                      'bg_tr': to_tensor(back),
                      'multi_fr': to_tensor(image_affine)}

            return sample

        except Exception as exception:
            print("Error Occurred While Loading: " + self.frames.iloc[item, 0])
            print(exception)


############################## Functions ##############################


def create_segmentation(alpha, trimap):
    """
    Creating the segmentation of the alpha based on the trimap
    """
    n = np.random.randint(0, 3)
    crop_sizes = [(15, 15), (25, 25), (35, 35), (45, 45)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    segmentation = (alpha > 0.5).astype(np.float32)

    segmentation = cv2.erode(segmentation, kernel=kernel, iterations=np.random.randint(10, 20))
    segmentation = cv2.dilate(segmentation, kernel=kernel, iterations=np.random.randint(15, 30))

    segmentation = segmentation.astype(np.float32)
    segmentation = (255 * segmentation).astype(np.uint8)

    for i in range(n):
        cropping = random.choice(crop_sizes)
        x, y = random_choice(trimap, cropping)
        segmentation = crop_holes(segmentation, x, y, cropping)
        trimap = crop_holes(trimap, x, y, cropping)

    k = [(21, 21), (31, 31), (41, 41)]
    segmentation = cv2.GaussianBlur(segmentation.astype(np.float32), random.choice(k), 0)

    return segmentation.astype(np.float32)


def create_segmentation_guide(segmentation, resolution, ksize=False):
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    segmentation = segmentation.astype(np.float32) / 255
    segmentation[segmentation > 0.2] = 1
    K = 25

    zero_ID = np.nonzero(np.sum(segmentation, axis=1) == 0)
    delete_ID = zero_ID[0][zero_ID[0] > 250]

    if len(delete_ID) > 0:
        delete_ID = [delete_ID[0] - 2, delete_ID[0] - 1, *delete_ID]
        segmentation = np.delete(segmentation, delete_ID, 0)

    segmentation = cv2.copyMakeBorder(segmentation, 0, K + len(delete_ID), 0, 0, cv2.BORDER_REPLICATE)

    segmentation = cv2.erode(segmentation, kernel_erode, iterations=np.random.randint(10, 20))
    segmentation = cv2.dilate(segmentation, kernel_dilate, iterations=np.random.randint(3, 7))

    k = [(21, 21), (31, 31), (41, 41)]
    if ksize:
        segmentation = cv2.GaussianBlur(segmentation.astype(np.float32), (31, 31), 0)
    else:
        segmentation = cv2.GaussianBlur(segmentation.astype(np.float32), random.choice(k), 0)
    segmentation = (255 * segmentation).astype(np.uint8)
    segmentation = np.delete(segmentation, range(resolution[0], resolution[0] + K), 0)

    return segmentation


def crop_holes(image, x, y, size):
    """
    Crop the given image from the location x, y of the given size
    """
    image[y:y+size[0], x:x+size[1]] = 0
    return image


def apply_crop(image, box, resolution):
    """
    Return a cropped image from the given image of the given size
    """
    image = image[box[0]: box[0] + box[2], box[1]: box[1] + box[3], ...]
    image = cv2.resize(image, resolution)
    return image


def create_bbox(mask, w, h):
    """
    Creating a bounding box for the image
    """
    loc = np.array(np.where(mask))
    x1, y1 = np.amin(loc, axis=1)
    x2, y2 = np.amax(loc, axis=1)

    min_w = np.maximum(y2 - y1, x2 - x1)
    boundary = np.random.uniform(0.1, 0.4)
    x1 = x1 - np.round(boundary * min_w)
    y1 = y1 - np.round(boundary * min_w)
    y2 = y2 + np.round(boundary * min_w)

    # Check for the boundaries
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= w:
        x2 = w - 1
    if y2 >= h:
        y2 = h - 1

    box = np.around([x1, y1, x2 - x1, y2 - y1]).astype('int')

    return box


def composite(foreground, background, alpha):
    """
    Create a composite image using the following equation:
    I = alpha * Fg + (1 - alpha) * Bg
    """
    # Convert to Float
    Fg = foreground.astype(np.float32)
    Bg = background.astype(np.float32)
    alpha = alpha.astype(np.float32)
    alpha = np.expand_dims(alpha / 255, axis=2)

    # Create Image
    I = alpha * Fg + (1 - alpha) * Bg
    I = I.astype(np.uint8)
    return I


def add_noise(background, mean, sigma):
    """
    Add Gaussian noise to the background with mean and sigma
    """
    # Generate Gaussian Noise
    background = background.astype(np.float32)
    gaussian_noise = np.random.normal(mean, sigma, background.shape)
    gaussian_noise = gaussian_noise.resize(background.shape)

    # Add the noise to the image
    noisy_image = background + gaussian_noise

    # Make the values of image with in the range
    noisy_image[noisy_image < 0] = 0
    noisy_image[noisy_image > 255] = 255

    return noisy_image


def safe_crop(image, x, y, crop_size, image_size):
    """
    Crop the image from x, y location of crop size
    """
    h, w = image_size
    crop_h, crop_w = crop_size

    # Create an empty matrix
    if len(image.shape) == 2:
        img = np.zeros((crop_h, crop_w), np.float32)
    else:
        img = np.zeros((crop_h, crop_w, 3), np.float32)

    # Copy the cropped image
    crop = img[y:y + crop_h, x:x + crop_w]
    img[0:crop.shape[0], 0:crop.shape[1]] = crop

    # Resize the cropped image
    if crop_size != (h, w):
        img = cv2.resize(img, dsize=(h, w))

    return img


def generate_trimap(alpha, lower_k, higher_k, train):
    """
    Generate the trimap for the alpha
    """
    # Create a black image and the kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = np.array(np.equal(alpha, 255).astype(np.float32))

    # Get the number of iterations based on mode i.e. whether train mode or not
    if train:
        k = np.random.randint(lower_k, higher_k)
    else:
        k = np.round((lower_k + higher_k) / 2).astype('int')

    # Apply the erode and dilate
    img = cv2.erode(img, kernel, iterations=k)
    temp = np.array(np.not_equal(alpha, 0).astype(np.float32))
    temp = cv2.dilate(temp, kernel, iterations=2 * k)

    # Generate Trimap
    trimap = img * 255 + (temp - img) * 128

    return trimap.astype(np.uint8)


def random_choice(trimap, crop=(320, 320)):
    """
    Generate a random choice of x and y within the crop size for the trimap
    """
    h, w = trimap.shape[0:2]
    crop_h, crop_w = crop

    value = np.zeros((h, w))
    value[(crop_h / 2):(h - crop_h / 2), (crop_w / 2):(w - crop_w / 2)] = 1

    y_list, x_list = np.where(np.logical_and(trimap == 128, value == 1))
    x, y = 0, 0

    if len(y_list) > 0:
        i = np.random.choice(range(len(y_list)))
        cx = x_list[i]
        cy = y_list[i]
        x = max(0, cx - int(crop_w / 2))
        y = max(0, cy - int(crop_h / 2))

    return x, y


def to_tensor(image):
    """
    Convert the image from numpy to tensor element
    """
    if len(image.shape) >= 3:
        img = torch.from_numpy(image.transpose((2, 0, 1)))
    else:
        img = torch.from_numpy(image)
        img = img.unsqueeze(0)

    return 2 * (img.float().div(255)) - 1

