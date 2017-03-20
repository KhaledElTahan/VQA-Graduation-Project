import numpy as np
from data_fetching.img_fetcher import get_img_batch
from data_fetching.question_fetcher import get_question_batch
from data_fetching.annotation_fetcher import get_annotation_batch

TRAIN_START_ID = 20000  # Starting image_id of the training data set
TRAIN_END_ID = 29999    # Ending image_id of the training data set
VAL_START_ID = 20000    # Starting image_id of the validation data set
VAL_END_ID = 29999      # Ending image_id of the validation data set


# returns a batch of the data set
def get_data_batch(itr, batch_size, training_data):

    start_id = TRAIN_START_ID if training_data else VAL_START_ID
    end_id = TRAIN_END_ID if training_data else VAL_END_ID

    data_len = end_id - start_id + 1
    itr = itr % data_len

    actual_batch_size = min(batch_size, data_len - itr)

    images = get_img_batch(start_id + itr, actual_batch_size, training_data)
    questions = get_question_batch(start_id + itr, actual_batch_size, training_data)
    annotations = get_annotation_batch(start_id + itr, actual_batch_size, training_data)

    # if we reached the end of data we continue the batch from the begining
    if actual_batch_size < batch_size:
        rem_batch_size = batch_size - actual_batch_size
        images = np.append(images, get_img_batch(start_id, rem_batch_size, training_data), axis=0)
        questions = np.append(questions, get_question_batch(start_id, rem_batch_size, training_data), axis=0)
        annotations = np.append(annotations, get_annotation_batch(start_id, rem_batch_size, training_data), axis=0)
    

    return images, questions, annotations


