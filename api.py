# from tests import tester
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/data_fetching')
sys.path.insert(0, 'src/feature_extraction')

from src.model import evaluate
from src.model import train_model
from src.data_fetching.data_fetcher import DataFetcher
from src.feature_extraction.img_features import extract
from src.data_fetching.annotation_fetcher import get_top_answers
from src.sentence_preprocess import question_batch_to_vecs

def run_tests(system_args):
    # tester.run_tests(system_args)
    pass

def train(model_name,
          batch_size,
          from_scratch_flag, 
          validate_flag, 
          trace_flag, 
          validation_itr, 
          checkpoint_itr, 
          number_of_iteration, 
          starting_training_point):

    learning_rate = 1e-4
    
    train_model(starting_training_point,
                number_of_iteration,
                checkpoint_itr, 
                validation_itr, 
                learning_rate,  
                batch_size, 
                from_scratch_flag, 
                validate_flag, 
                trace_flag)

def evaluate_example(image, question, model_name):
    image_features = extract(image)
    question_features, words_count = question_batch_to_vecs([question])
    evaluation_logits = evaluate(image_features, question_features, words_count)
    answer_index = evaluation_logits.index(max(evaluation_logits))
    top_answers = get_top_answers()
    return top_answers[answer_index]

def terminate_evaluation(model_name):
    pass

def validate_system(batch_size, data_path, model_name, validation_size):
    pass

def test_model(batch_size, data_path, model_name, test_size):
    pass

def _set_data_paths(train_data_path, validate_data_path):
    set_images_data_path(train_data_path, validate_data_path)
    set_annotations_data_path(train_data_path, validate_data_path)
    set_questions_data_path(train_data_path, validate_data_path)


train("7amada", 32, True, True, True, 0, 0, 10, 0)