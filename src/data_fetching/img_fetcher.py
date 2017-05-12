import skimage.io
from skimage import transform
from data_fetching.multithreading import FuncThread
import math
import numpy as np
import glob

# Directories assume that Process Working Directory is the src folder
TRAIN_SET_DIR = "../data/scene_img_abstract_v002_val2015/"  # Change it to the large data set later
VAL_SET_DIR = "../data/scene_img_abstract_v002_val2015/"
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


# Returns a numpy array containing images and a boolean which is true if we reached the end of the data set
def _get_img_batch(img_dir, start_id, batch_size):
    batch = []

    for i in range(start_id, start_id + batch_size):
        img = _get_img_by_id(img_dir, i)
        batch.append(img)

    batch_np = np.stack(batch, axis=0)

    return batch_np


# Overloaded for data set type and multi-threading
def get_img_batch(start_id, batch_size, training_data):

    num_threads = math.ceil(batch_size / IMG_PER_THREAD)
    img_threads = []

    for i in range(0, num_threads):
        img_threads.append(FuncThread(_get_img_batch, TRAIN_SET_DIR if training_data else VAL_SET_DIR,
                                      start_id + i * IMG_PER_THREAD,
                                      min(IMG_PER_THREAD, batch_size - i * IMG_PER_THREAD)))

    batch = img_threads[0].get_ret_val()

    for i in range(1, num_threads):
        batch = np.concatenate((batch, img_threads[i].get_ret_val()), axis=0)

    return batch