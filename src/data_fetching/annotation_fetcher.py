import json
import numpy as np
import operator

# Directories assume that Process Working Directory is the src folder
TRAIN_SET_JSON = "../data/abstract_v002_val2015_annotations.json"	    # Change it to the large data set later
VAL_SET_JSON = "../data/abstract_v002_val2015_annotations.json"

TRAIN_SET_ANNOTATIONS = None
VAL_SET_ANNOTATIONS = None

TOP_ANSWERS = None
TOP_ANSWERS_MAP = None      # Key is answer string, Value is its index in the TOP_ANSWERS array
TOP_ANSWERS_COUNT = 1000

# Loads the json file and saves it in the global variable as a dictionary of
# Key = question_id and
# Value = list of answers
def _load_json_file(file_name):
    a_dict = {}
    with open(file_name) as data_file:
        data = json.load(data_file)
        annotations = data["annotations"]

    for elem in annotations:
        a_dict[elem["question_id"]] = [(answer["answer"],answer["answer_confidence"]) for answer in elem["answers"]]
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
        for q_id in range(0,3):
            question_id = img_id * 10 + q_id
            batch.append(expand_answer(all_annotation[question_id]))

    batch_np = np.stack(batch, axis=0)

    return batch_np


# Takes a list of answers_pair where first item is the answer string and second item is the answer confidence
# Returns a vector of length = TOP_ANSWERS_COUNT
def expand_answer(answers_pair):

    if TOP_ANSWERS_MAP is None:
        TOP_ANSWERS = get_top_answers()

    answers_dict = {}
    expanded_answer = [0] * TOP_ANSWERS_COUNT
    total_sum = 0

    for elem in answers_pair:

        ans = elem[0]
        conf = elem[1]

        if ans in TOP_ANSWERS_MAP:
            if conf == "yes":
                if ans in answers_dict:
                    answers_dict[ans] += 2
                else:
                    answers_dict[ans] = 2
                total_sum += 2

            if conf == "maybe":
                if ans in answers_dict:
                    answers_dict[ans] += 1
                else:
                    answers_dict[ans] = 1
                total_sum += 1

    for answer in answers_dict:
        answers_dict[answer] = answers_dict[answer] / total_sum

    for key, val in answers_dict.items():
        expanded_answer[TOP_ANSWERS_MAP[key]] = val

    return expanded_answer


# Returns the top answers 
def get_top_answers():

    global TOP_ANSWERS, TOP_ANSWERS_MAP

    if TOP_ANSWERS is not None:
        return TOP_ANSWERS

    top_answers_dict = {}

    annotations = _get_annotations(True);

    for key, answers in annotations.items():

        for answer in answers:

            ans = answer[0]
            conf = answer[1]

            if conf == "yes":
                if ans in top_answers_dict:
                    top_answers_dict[ans] += 2
                else:
                    top_answers_dict[ans] = 2
            elif conf == "maybe":
                if ans in top_answers_dict:
                    top_answers_dict[ans] += 1
                else:
                    top_answers_dict[ans] = 1
            else :
                if not (ans in top_answers_dict):
                    top_answers_dict[ans] = 0
                
        
    #return 2 columns array sorted on the second column, the first column is the answer and the second column is the count
    sorted_top_answers = sorted(top_answers_dict.items(), key=operator.itemgetter(1), reverse = True) 
    #return the first column of the sorted_top_answers, that are the words

    if TOP_ANSWERS_COUNT > len(sorted_top_answers):
        raise ValueError("Top answers count is more than the number of answers !\n TOP_ANSWERS_COUNT = " + TOP_ANSWERS_COUNT)

    TOP_ANSWERS = ([row[0] for row in sorted_top_answers[:TOP_ANSWERS_COUNT]])

    TOP_ANSWERS_MAP = {}

    for i in range(len(TOP_ANSWERS)):
        ans = TOP_ANSWERS[i]
        TOP_ANSWERS_MAP[ans] = i;

    return TOP_ANSWERS

def set_annotations_data_path(train_data_path, validate_data_path):
    TRAIN_SET_JSON = train_data_path
    VAL_SET_JSON = validate_data_path