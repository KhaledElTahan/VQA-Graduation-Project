import numpy as np
from data_fetching.img_fetcher import get_imgs_batch
from data_fetching.question_fetcher import get_questions_batch, get_questions_len
from data_fetching.annotation_fetcher import get_annotations_batch
from data_fetching.multithreading import FuncThread
from data_fetching.data_path import get_path
from sentence_preprocess import question_batch_to_vecs
from feature_extraction import img_features


class DataFetcher:

    def __init__(self, evaluation_type, batch_size=32, start_itr=0):

        self.evaluation_type = evaluation_type
        self.batch_size = batch_size
        self.itr = start_itr

        self.available_datasets = ['abstract_scenes_v1', 'abstract_scenes_v1']

        self.data_lengthes = [self.get_dataset_len(dataset_name) for dataset_name in self.available_datasets]
        self.sum_data_len = sum(self.data_lengthes)


    #  Returns the name of the current dataset
    def get_current_dataset(self):

        itr = self.itr % self.sum_data_len

        idx = 0
        for i in range(len(self.data_lengthes)):

            if itr >= self.data_lengthes[i]:
                itr -= self.data_lengthes[i]
                idx = (idx + 1) % len(self.data_lengthes)

        return self.available_datasets[idx]

    # Returns the iterator of the current dataset
    def get_dataset_iterator(self):

        itr = self.itr % self.sum_data_len

        for i in range(len(self.data_lengthes)):

            if itr >= self.data_lengthes[i]:
                itr -= self.data_lengthes[i]

        return itr

    # Return path to images of the current dataset
    def get_img_path(self):
        return get_path(self.evaluation_type, self.get_current_dataset(), 'images')

    # Return path to questions of the current dataset
    def get_questions_path(self):
        return get_path(self.evaluation_type, self.get_current_dataset(), 'questions')

    # Return path to annotations of the current dataset
    def get_annotations_path(self):
        return get_path(self.evaluation_type, self.get_current_dataset(), 'annotations')

    # Extract features from images and return a dictionary { "image_id": features }
    def images_to_features(self, images_dict):

        image_ids, batch = list(images_dict.keys()), list(images_dict.values())
        features = img_features.extract(batch)

        for i in range(len(features)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            images_dict[image_ids[i]] = features[i]

        return images_dict

    # Link questions with images and annotations using ids
    def merge_by_id(self, questions_all, annotations_dict, images_dict):

        annotations, images = [], []

        for q in questions_all:

            annotations.append(annotations_dict[q["question_id"]])
            images.append(images_dict[q["image_id"]])

        return np.array(annotations), np.array(images)

    def questions_to_features(self, questions_all):

        questions = [q["question"] for q in questions_all]

        questions_vecs, questions_length = question_batch_to_vecs(questions)

        return questions_vecs, questions_length

    def load_images(self, questions_all):

        # Extract image ids
        image_ids = list(set([elem['image_id'] for elem in questions_all]))

        # Load images
        images_dict = get_imgs_batch(image_ids, self.get_img_path())

        # Extract features from images
        images_dict = self.images_to_features(images_dict)

        return images_dict

    def load_annotations(self, questions_all):

        # Extract question ids
        question_ids = [elem['question_id'] for elem in questions_all]

        # Load annotations
        annotations_dict = get_annotations_batch(question_ids, self.get_annotations_path())

        return annotations_dict

    # Updates state of the loader to prepare for the next batch
    def update_state(self, actual_batch_size):

        self.itr += actual_batch_size

    def get_dataset_len(self, dataset_name):
        path = get_path(self.evaluation_type, dataset_name, 'questions')
        return get_questions_len(path)

    def _get_next_batch(self, batch_size):

        questions_all = get_questions_batch(self.get_dataset_iterator(), batch_size, self.get_questions_path())

        # Extract features from questions
        question_features_thread = FuncThread(self.questions_to_features, questions_all)
        
        # Load and extract features from images
        images_dict_thread = FuncThread(self.load_images, questions_all)

        # Load annotations
        annotations_dict_thread = FuncThread(self.load_annotations, questions_all)

        # Link questions with images and annotations using ids
        annotations, images = self.merge_by_id(questions_all, annotations_dict_thread.get_ret_val(), images_dict_thread.get_ret_val())

        questions_vecs, questions_length = question_features_thread.get_ret_val()

        # Updates state of the loader to prepare for the next batch
        self.update_state(len(questions_all))

        return images, questions_vecs, questions_length, annotations

    def get_next_batch(self):

        images, questions_vecs, questions_length, annotations = self._get_next_batch(self.batch_size)

        actual_batch_size = len(images)

        # read all datasets and starting from the begining
        eof = (self.itr % self.sum_data_len) == 0

        if actual_batch_size < self.batch_size:

            images_2, questions_vecs_2, questions_length_2, annotations_2 = self._get_next_batch(self.batch_size - actual_batch_size)

            images = np.append(images, images_2, axis=0)
            questions_vecs = np.append(questions_vecs, questions_vecs_2, axis=0)
            questions_length = np.append(questions_length, questions_length_2, axis=0)
            annotations = np.append(annotations, annotations_2, axis=0)

        return images, questions_vecs, questions_length, annotations, eof


