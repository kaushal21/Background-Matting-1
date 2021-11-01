import numpy as np
import torchvision
import cv2


def composing_image(fore_ground, back_ground, alpha):
    """
    Get the image using fore ground, background and the alpha using the following equation
    I = alpha * F + (1 - alpha) * B
    """
    fore_ground = np.array(fore_ground, np.float32)
    alpha = np.expand_dims(alpha / 255, axis=2)
    image = alpha * fore_ground + (1 - alpha) * back_ground
    image = image.astype(np.uint8)
    return image


def get_bounding_box(mask, A, B):
    """
    Get the bounding box for the mask of shape A x B
    """
    location = np.array(np.where(mask))
    x1, y1 = np.amin(location, axis=1)
    x2, y2 = np.amax(location, axis=1)

    box = [x1, y1, np.maximum(x2 - x1, y2 - y1), np.maximum(x2 - x1, y2 - y2)]
    box = create_bounding_box(box, (A, B))

    return box


def create_bounding_box(box, shape):
    """
    Calculate the bounding box within the shape of the image.
    """
    weight = np.maximum(box[2], box[3])

    x1 = box[0] - 0.1 * weight
    y1 = box[1] - 0.1 * weight
    x2 = box[0] + 1.1 * weight
    y2 = box[1] + 1.1 * weight

    # If the point is outside the shape, then change it to an end point of the shape
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 >= shape[0]:
        x2 = shape[0] - 1
    if y2 >= shape[1]:
        y2 = shape[1] - 1

    bounding_box = np.around([x1, y1, x2 - x1, y2 - y1]).astype('int')

    return bounding_box


def crop_image(image, size, box):
    """
    Resize the image passed to the given size
    """
    if image.ndim >= 3:
        image = image[box[0]: box[0] + box[2], box[1]: box[1] + box[3], ...]
        image = cv2.resize(image, size)
    else:
        image = image[box[0]: box[0] + box[2], box[1]: box[1] + box[3]]
        image = cv2.resize(image, size)

    return image


def un_crop(image, box, A=720, B=1280):
    """
    Uncrop the image i.e., Resize the image and pad it with zeros
    """
    image = cv2.resize(image, (box[3], box[2]))

    if image.ndim == 2:
        unCropped = np.zeros((A, B))
        unCropped[box[0]: box[0] + box[2], box[1]: box[1] + box[3]] = image
    else:
        unCropped = np.zeros((A, B, 3))
        unCropped[box[0]: box[0] + box[2], box[1]: box[1] + box[3], :] = image

    return unCropped


def convert_to_image(p):
    """
    Get the image for the passed on data i.e., maybe the alpha matte or the segmentation
    """
    data = ((p.data).cpu()).numpy()
    data = (data + 1) / 2
    image = data.transpose((1, 2, 0))
    image[image > 1] = 1
    image[image < 0] = 0

    return image

