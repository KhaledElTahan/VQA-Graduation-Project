import numpy as np
from data_fetching.img_fetcher import get_img_batch
from data_fetching.question_fetcher import get_question_batch
from data_fetching.annotation_fetcher import get_annotation_batch
from data_fetching.multithreading import FuncThread

TRAIN_START_ID = 20000  # Starting image_id of the training data set
TRAIN_END_ID = 29999    # Ending image_id of the training data set
VAL_START_ID = 20000    # Starting image_id of the validation data set
VAL_END_ID = 29999      # Ending image_id of the validation data set

# returns a batch of the data set
def get_data_batch(itr, batch_size, training_data):

    start_id = TRAIN_START_ID if training_data else VAL_START_ID
    end_id = TRAIN_END_ID if training_data else VAL_END_ID

    data_len = end_id - start_id + 1
    itr %= data_len

    actual_batch_size = min(batch_size, data_len - itr)

    img_thread = FuncThread(get_img_batch, start_id + itr, actual_batch_size, training_data)
    question_thread = FuncThread(get_question_batch, start_id + itr, actual_batch_size, training_data)
    annotation_thread = FuncThread(get_annotation_batch, start_id + itr, actual_batch_size, training_data)

    images = None
    questions = None
    annotations = None

    # if we reached the end of data we continue the batch from the begining
    if actual_batch_size < batch_size:
        rem_batch_size = batch_size - actual_batch_size

        rem_img_thread = FuncThread(get_img_batch, start_id, rem_batch_size, training_data)
        rem_question_thread = FuncThread(get_question_batch, start_id, rem_batch_size, training_data)
        rem_annotation_thread = FuncThread(get_annotation_batch, start_id, rem_batch_size, training_data)

        images = np.append(img_thread.get_ret_val(), rem_img_thread.get_ret_val(), axis=0)
        questions = np.append(question_thread.get_ret_val(), rem_question_thread.get_ret_val(), axis=0)
        annotations = np.append(annotation_thread.get_ret_val(), rem_annotation_thread.get_ret_val(), axis=0)

    else:

        images = img_thread.get_ret_val()
        questions = question_thread.get_ret_val()
        annotations = annotation_thread.get_ret_val()

    return images, questions, annotations
