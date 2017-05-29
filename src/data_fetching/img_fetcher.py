import skimage.io
from skimage import transform
from data_fetching.multithreading import FuncThread
import math
import numpy as np
import glob
import pickle

IMG_SHAPE = (224, 224)  # Used image shape for resizing
IMG_PER_THREAD = 8  # Number of images loaded per thread determined by trial and error according to the batch size

# Returns resized numpy array of the image with this id
def _get_img_by_id(img_dir, img_id):
    suffix = "*" + format(img_id, '012d') + ".png"
    files = glob.glob(img_dir + suffix)

    if len(files) == 0:
        raise ValueError("No image found with suffix = " + suffix)

    img = skimage.io.imread(files[0])
    img = transform.resize(img, IMG_SHAPE)

    # Modifications to the image if it's a grayscale or contains an alpha channel
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)

    return np.asarray(img) * 255.0

def _get_img_feature_by_id(img_dir, img_id):
    file_name = format(img_id, '012d') + ".bin"
    files = glob.glob(img_dir + file_name)

    if len(files) == 0:
        raise ValueError("No image feature found with name = " + file_name)

    with open (files[0], 'rb') as fp:
        features = pickle.load(fp)

    return features


# Returns a numpy array containing images and a boolean which is true if we reached the end of the data set
def _get_imgs_batch(img_dir, image_ids):

    batch = {}

    for id in image_ids:
        img = _get_img_by_id(img_dir, id)
        batch[id] = img

    return batch

def _get_imgs_feature_batch(img_dir, image_ids):

    batch = {}

    for id in image_ids:
        img = _get_img_feature_by_id(img_dir, id)
        batch[id] = img

    return batch

# Overloaded for data set type and multi-threading
def get_imgs_batch(image_ids, img_dir):

    # num_threads = math.ceil(len(image_ids) / IMG_PER_THREAD)
    # img_threads = []

    # for i in range(0, num_threads):

    #     ids_slice = image_ids[0: min(IMG_PER_THREAD, len(image_ids))]
    #     image_ids = image_ids[len(ids_slice):]

    #     img_threads.append(FuncThread(_get_imgs_batch, img_dir, ids_slice))

    #batch = {}

    # for i in range(0, num_threads):
    #     batch = {**batch, **img_threads[i].get_ret_val()}

    return _get_imgs_batch(img_dir, image_ids)

# Return features of an image
def get_imgs_features_batch(image_ids, img_dir):

    return _get_imgs_feature_batch(img_dir, image_ids)