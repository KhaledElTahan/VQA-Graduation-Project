
import tests_basis
from VQA.src.f_extractor import get_features
import skimage.io
from skimage import transform
import numpy as np

img = skimage.io.imread(tests_basis.get_test_image_path('resnet_test.jpg'))
resized_img1 = transform.resize(img, (224, 224))
resized_img2 = transform.resize(img, (224, 224))

batch = np.array([resized_img1, resized_img2])

print(get_features(batch).shape)

