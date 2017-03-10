import json
import numpy as np

# Directories assume that Process Working Directory is the src folder
TRAIN_SET_JSON = "../data/abstract_v002_val2015_annotations.json"	    # Change it to the large data set later
VAL_SET_JSON = "../data/abstract_v002_val2015_annotations.json"

TRAIN_SET_ANNOTATIONS = None
VAL_SET_ANNOTATIONS = None


# Loads the json file and saves it in the global variable as a dictionary of
# Key = question_id and
# Value = list of answers
def _load_json_file(file_name):
    a_dict = {}
    with open(file_name) as data_file:
        data = json.load(data_file)
        annotations = data["annotations"]

    for elem in annotations:
        a_dict[elem["question_id"]] = [answers["answer"] for answers in elem["answers"]]
    return a_dict


# Returns the global dictionary for annotations based on the type of the data set
def _get_annotations(training_data):

    global TRAIN_SET_ANNOTATIONS, VAL_SET_ANNOTATIONS

    if training_data:
        if TRAIN_SET_ANNOTATIONS is None:
            TRAIN_SET_ANNOTATIONS = _load_json_file(TRAIN_SET_JSON)
        return TRAIN_SET_ANNOTATIONS
    else:
        if VAL_SET_ANNOTATIONS is None:
            VAL_SET_ANNOTATIONS = _load_json_file(VAL_SET_JSON)
        return VAL_SET_ANNOTATIONS


# Returns a batch of the annotations given the start id of the image, batch size and the type of the data set
# The shape of the returned numpy array is (batch_size, 3, 10)
# 3 is the number of questions for every image
# 10 is the number of answers for each question
def get_annotation_batch(start_id, batch_size, training_data):

    all_annotation = _get_annotations(training_data)
    batch = []

    for img_id in range(start_id, start_id + batch_size):
        q_annots = []

        for q_id in range(0,3):
            question_id = img_id * 10 + q_id
            q_annots.append(all_annotation[question_id])

        batch.append(q_annots)

    batch_np = np.stack(batch, axis=0)

    return batch_np
