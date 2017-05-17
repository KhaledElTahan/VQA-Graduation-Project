import json
import numpy as np

# Directories assume that Process Working Directory is the src folder
TRAIN_SET_JSON = "../data/OpenEnded_abstract_v002_val2015_questions.json"	    # Change it to the large data set later
VAL_SET_JSON = "../data/OpenEnded_abstract_v002_val2015_questions.json"

TRAIN_SET_QUESTIONS = None
VAL_SET_QUESTIONS = None


# Loads the json file and saves it in the global variable as a dictionary of
# Key = question_id
# Value = question
def _load_json_file(file_name):
    q_dict = {}
    with open(file_name) as data_file:
        data = json.load(data_file)
        questions = data["questions"]

    for elem in questions:
        q_dict[elem["question_id"]] = elem["question"]
    return q_dict


# Returns the global dictionary for questions based on the type of the data set
def _get_questions(training_data):

    global TRAIN_SET_QUESTIONS, VAL_SET_QUESTIONS

    if training_data:
        if TRAIN_SET_QUESTIONS is None:
            TRAIN_SET_QUESTIONS = _load_json_file(TRAIN_SET_JSON)
        return TRAIN_SET_QUESTIONS
    else:
        if VAL_SET_QUESTIONS is None:
            VAL_SET_QUESTIONS = _load_json_file(VAL_SET_JSON)
        return VAL_SET_QUESTIONS


# Returns a batch of the questions given the start id of the image, batch size and the type of the data set
# The shape of the returned numpy array is (batch_size, 3) where 3 is the number of questions for every image
def get_question_batch(start_id, batch_size, training_data):

    all_questions = _get_questions(training_data)
    batch = []

    for img_id in range(start_id, start_id + batch_size):
        batch.append(all_questions[img_id * 10])
        batch.append(all_questions[img_id * 10 + 1])
        batch.append(all_questions[img_id * 10 + 2])

    batch_np = np.stack(batch, axis=0)

    return batch_np

def set_questions_data_path(train_data_path, validate_data_path):
    global TRAIN_SET_JSON, VAL_SET_JSON
    TRAIN_SET_JSON = train_data_path
    VAL_SET_JSON = validate_data_path
