import json
import numpy as np
import operator
from data_fetching.data_path import get_path,get_top_answers_path
import pickle
import os.path

TOP_ANSWERS_PATH = get_top_answers_path()
TOP_ANSWERS_MAP = None      # Key is answer string, Value is its index in the TOP_ANSWERS array
TOP_ANSWERS_LIST = None
TOP_ANSWERS_COUNT = 1000

LOADED_JSON_FILES = {}

# Loads the json file and saves it in the global variable as a dictionary of
# Key = question_id and
# Value = list of answers
def _load_json_file(file_name):

    a_dict = {}
    with open(file_name) as data_file:
        data = json.load(data_file)
        annotations = data["annotations"]

    for elem in annotations:
        a_dict[elem["question_id"]] = [(answer["answer"],
            answer["answer_confidence"] if "answer_confidence" in answer else "yes")
             for answer in elem["answers"]]
             
    return a_dict


# Returns the global dictionary for annotations based on the type of the data set
def _get_annotations(file_name):

    global LOADED_JSON_FILES

    if file_name in LOADED_JSON_FILES:
        return LOADED_JSON_FILES[file_name]
    else:
        LOADED_JSON_FILES[file_name] = _load_json_file(file_name)
        return LOADED_JSON_FILES[file_name]


# Returns a batch of the annotations given the start id of the image, batch size and the type of the data set
# The shape of the returned numpy array is (batch_size, 3, 10)
# 3 is the number of questions for every image
# 10 is the number of answers for each question
def get_annotations_batch(question_ids, file_name):

    all_annotation = _get_annotations(file_name)
    batch = {}

    for q_id in question_ids:
            batch[q_id] = expand_answer(all_annotation[q_id])

    return batch


# Takes a list of answers_pair where first item is the answer string and second item is the answer confidence
# Returns a vector of length = TOP_ANSWERS_COUNT
def expand_answer(answers_pair):

    global TOP_ANSWERS_MAP, TOP_ANSWERS_LIST

    if TOP_ANSWERS_MAP is None:
        TOP_ANSWERS_MAP, TOP_ANSWERS_LIST = get_top_answers_map()

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


def get_top_answers():
    if TOP_ANSWERS_MAP is None:
        TOP_ANSWERS_MAP, TOP_ANSWERS_LIST = get_top_answers_map()

    return TOP_ANSWERS_LIST

# Returns the top answers 
def get_top_answers_map():

    global TOP_ANSWERS_MAP, TOP_ANSWERS_LIST

    if os.path.exists(TOP_ANSWERS_PATH):

        with open (TOP_ANSWERS_PATH, 'rb') as fp:
            TOP_ANSWERS_MAP, TOP_ANSWERS_LIST = pickle.load(fp)

        return TOP_ANSWERS_MAP, TOP_ANSWERS_LIST

    top_answers_dict = {}

    annotations_abstract_v1 = _get_annotations(get_path('training', 'abstract_scenes_v1', 'annotations'))
    annotations_balanced_binary_abstract = _get_annotations(get_path('training', 'balanced_binary_abstract_scenes', 'annotations'))
    annotations_balanced_real = _get_annotations(get_path('training', 'balanced_real_images', 'annotations'))

    all_annotations = [annotations_abstract_v1, annotations_balanced_binary_abstract, annotations_balanced_real]

    for annot_dict in all_annotations:
        for key, answers in annot_dict.items():

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
                else:
                    if not (ans in top_answers_dict):
                        top_answers_dict[ans] = 0
                   
    # return 2 columns array sorted on the second column, the first column is the answer and the second column is the count
    sorted_top_answers = sorted(top_answers_dict.items(), key=operator.itemgetter(1), reverse=True) 
    # return the first column of the sorted_top_answers, that are the words

    if TOP_ANSWERS_COUNT > len(sorted_top_answers):
        raise ValueError("Top answers count is more than the number of answers !\n TOP_ANSWERS_COUNT = " + TOP_ANSWERS_COUNT)

    TOP_ANSWERS_LIST = ([row[0] for row in sorted_top_answers[:TOP_ANSWERS_COUNT]])

    TOP_ANSWERS_MAP = {}

    for i in range(len(TOP_ANSWERS_LIST)):
        ans = TOP_ANSWERS_LIST[i]
        TOP_ANSWERS_MAP[ans] = i

    # Write map and list to file
    with open(TOP_ANSWERS_PATH, 'wb') as fp:
        pickle.dump([TOP_ANSWERS_MAP, TOP_ANSWERS_LIST], fp)

    return TOP_ANSWERS_MAP, TOP_ANSWERS_LIST
