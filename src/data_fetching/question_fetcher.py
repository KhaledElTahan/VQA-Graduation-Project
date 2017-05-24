import json
import numpy as np

LOADED_JSON_FILES = {}

# Loads the json file and saves it in the global variable as a dictionary of
# Key = question_id
# Value = { image_id, question }
def _load_json_file(file_name):
    q_dict = {}
    with open(file_name) as data_file:
        data = json.load(data_file)
        questions = data["questions"]

        questions = sorted(questions, key=lambda k: k['image_id'])
        
    return questions


# Returns the global dictionary for questions based on the file_name
def _get_questions(file_name):

    global LOADED_JSON_FILES

    if file_name in LOADED_JSON_FILES:
        return LOADED_JSON_FILES[file_name]
    else:
        LOADED_JSON_FILES[file_name] = _load_json_file(file_name)
        return LOADED_JSON_FILES[file_name]

# Returns a batch of the questions given the start index of the question, batch size and the file_name of the dataset
# The return value is a list of dictionary objects each has image_id, question_id and question
def get_questions_batch(start, batch_size, file_name):

    all_questions = _get_questions(file_name)
    batch = []

    actual_batch_size = min(len(all_questions) - start, batch_size)

    for i in range(start, start + actual_batch_size):
        batch.append(all_questions[i])

    return batch

def get_questions_len(file_name):

    all_questions = _get_questions(file_name)
    return len(all_questions)
