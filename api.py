from tests import tester, tests_basis

def run_tests(system_args):
    tests_basis.set_options(system_args)
    tester.main(system_args)

def train_model(batch_size, data_path, validate_flag, validation_itr, checkpoint_itr, from_scratch, model_name, starting_training_point):
    pass

def evaluate_example(image, question, model_name):
    pass

def terminate_evaluation(model_name):
    pass

def validate_system(batch_size, data_path, model_name, validation_size):
    pass

def test_model(batch_size, data_path, model_name, test_size):
    pass
