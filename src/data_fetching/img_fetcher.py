import skimage.io
from skimage import transform
import numpy as np
import glob


# Directories assume that Process Working Directory is the src folder
TRAIN_SET_DIR = "../data/scene_img_abstract_v002_val2015/"		# Change it to the large data set later
VAL_SET_DIR = "../data/scene_img_abstract_v002_val2015/"
IMG_SHAPE = (224, 224)		# Used image shape for resizing


# Returns resized numpy array of the image with this id
def _get_img_by_id(img_dir, img_id):
    suffix = "*" + format(img_id, '012d') + ".png"
    files = glob.glob(img_dir + suffix)

    if len(files) == 0:
        raise ValueError("No image found with suffix = " + suffix)

    img = skimage.io.imread(files[0])
    img = transform.resize(img, IMG_SHAPE)
    return img


# Returns a numpy array containing images and a boolean which is true if we reached the end of the data set
def _get_img_batch(img_dir, start_id, batch_size):

    batch = []

    for i in range(start_id, start_id + batch_size):
        img = _get_img_by_id(img_dir, i)
        batch.append(img)

    batch_np = np.stack(batch, axis=0)

    return batch_np


# Overloaded for data set type
def get_img_batch(start_id, batch_size, training_data):

    if training_data:
        return _get_img_batch(TRAIN_SET_DIR,start_id,batch_size)
    else:
        return _get_img_batch(VAL_SET_DIR,start_id,batch_size)
