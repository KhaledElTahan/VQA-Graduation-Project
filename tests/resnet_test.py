import tests_basis
import sys
from f_extractor import get_features
import skimage.io
from skimage import transform
import numpy as np

def test_fn():
    img = skimage.io.imread(tests_basis.get_test_image_path('resnet_test.jpg'))
    resized_img1 = transform.resize(img, (224, 224))
    resized_img2 = transform.resize(img, (224, 224))

    batch = np.array([resized_img1, resized_img2])

    return get_features(batch).shape

def main(starting_counter):
    tests_basis.create_tests([test_fn], [None], [(2, 2048)])
    return tests_basis.main_tester("Testing the feature extraction from the ResNet-152L", starting_counter)

if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
