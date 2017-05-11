from img_features_mxnet import get_features

def extract(image_batch):
    """ 
        Requires batch of 3x224x224 Images
        Requires cells of range 0~225
        Example on how to preprocess the image using skimago.io

        def get_image(file_name):
            img = skimage.io.imread(file_name)
            img = transform.resize(img, (224, 224))
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            return np.asarray(img) * 255.0

        img_features.extract([get_image('dog'), get_image('cat')])
    """
    return get_features(image_batch)
