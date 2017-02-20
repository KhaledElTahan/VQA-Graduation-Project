
import tests_basis
from VQA.src.f_extractor import get_features
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
    test_1_res = tests_basis.test(test_fn, expected_output=(2, 2048))
    tests_results = [test_1_res]
    return tests_basis.main_tester("Testing the feature extraction from the ResNet-152L", starting_counter, tests_results)

if __name__ == "__main__":
    main(1)
