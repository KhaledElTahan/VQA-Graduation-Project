import tests_basis
import sys
from feature_extraction.img_features import extract
import skimage.io
from skimage import transform
import numpy as np
import os

def get_image(file_name):
    """ Required Image PreProcessing for the MXNet Model """
    img = skimage.io.imread(tests_basis.get_test_image_path(file_name))
    img = transform.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return np.asarray(img) * 255.0

def test_fn(size):
    img = get_image('resnet_test.jpg')
    os.system("nvidia-smi")
    return extract([img] * size).shape

def main(starting_counter):
    test_args, test_exps, test_fns = [8, 7, 6, 5, 4, 3, 2, 1] * 2, [], []

    test_fns.append(test_fn)
    test_exps.append((8, 2048))

    test_fns.append(test_fn)
    test_exps.append((7, 2048))

    test_fns.append(test_fn)
    test_exps.append((6, 2048))

    test_fns.append(test_fn)
    test_exps.append((5, 2048))

    test_fns.append(test_fn)
    test_exps.append((4, 2048))

    test_fns.append(test_fn)
    test_exps.append((3, 2048))

    test_fns.append(test_fn)
    test_exps.append((2, 2048))

    test_fns.append(test_fn)
    test_exps.append((1, 2048))

    test_fns.append(test_fn)
    test_exps.append((8, 2048))

    test_fns.append(test_fn)
    test_exps.append((7, 2048))

    test_fns.append(test_fn)
    test_exps.append((6, 2048))

    test_fns.append(test_fn)
    test_exps.append((5, 2048))

    test_fns.append(test_fn)
    test_exps.append((4, 2048))

    test_fns.append(test_fn)
    test_exps.append((3, 2048))

    test_fns.append(test_fn)
    test_exps.append((2, 2048))

    test_fns.append(test_fn)
    test_exps.append((1, 2048))

    tests_basis.create_tests(test_fns, test_args, test_exps)
    return tests_basis.main_tester("Testing rebinding resnet152-11k-mxnet", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
    