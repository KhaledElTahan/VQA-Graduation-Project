import os

data_fetching_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.abspath(os.path.join(data_fetching_path, os.pardir))
VQA_path = os.path.abspath(os.path.join(src_path, os.pardir))
data_path = os.path.join(VQA_path, "data")
VQA_Dataset_path = os.path.join(data_path, "VQA_Dataset")
training_path = os.path.join(VQA_Dataset_path, "training")
validation_path = os.path.join(VQA_Dataset_path, "validation")
testing_path = os.path.join(VQA_Dataset_path, "testing")

def get_path(evaluation_type, data_set_type, data_type):
    if evaluation_type == 'training':
        if data_set_type == 'balanced_real_images':
            if data_type == 'annotations':
                pass
            elif data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
            elif data_type == 'complementary_pairs_list':
                pass
        elif data_set_type == 'balanced_binary_abstract_scenes':
            if data_type == 'annotations':
                pass
            elif data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
        elif data_set_type == 'abstract_scenes_v1':
            if data_type == 'annotations':
                pass
            elif data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
    elif evaluation_type == 'validation':
        if data_set_type == 'balanced_real_images':
            if data_type == 'annotations':
                pass
            elif data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
            elif data_type == 'complementary_pairs_list':
                pass
        elif data_set_type == 'balanced_binary_abstract_scenes':
            if data_type == 'annotations':
                pass
            elif data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
        elif data_set_type == 'abstract_scenes_v1':
            if data_type == 'annotations':
                pass
            elif data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
    elif evaluation_type == 'testing':
        if data_set_type == 'balanced_real_images':
            if data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
        elif data_set_type == 'abstract_scenes_v1':
            if data_type == 'questions':
                pass
            elif data_type == 'images':
                pass
