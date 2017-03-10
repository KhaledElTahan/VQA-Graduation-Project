import numpy as np
from img_fetcher import get_img_batch
from question_fetcher import get_question_batch
from annotation_fetcher import get_annotation_batch

TRAIN_START_ID = 20000  # Starting image_id of the training data set
TRAIN_END_ID = 29999    # Ending image_id of the training data set
VAL_START_ID = 20000    # Starting image_id of the validation data set
VAL_END_ID = 29999      # Ending image_id of the validation data set


# returns a batch of the data set
def get_data_batch(itr, batch_size, training_data):

    start_id = TRAIN_START_ID if training_data else VAL_START_ID
    end_id = TRAIN_END_ID if training_data else VAL_END_ID

    actual_batch_size = min(batch_size, (end_id - start_id + 1) - itr)

    images = get_img_batch(start_id + itr, actual_batch_size, training_data)
    questions = get_question_batch(start_id + itr, actual_batch_size, training_data)
    annotations = get_annotation_batch(start_id + itr, actual_batch_size, training_data)

    print(images.shape)
    print(questions.shape)
    print(annotations.shape)

    # Kammelo hena ba2a :D

