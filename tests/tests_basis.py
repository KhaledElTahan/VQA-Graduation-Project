import sys
import os

sys.path.insert(0, '../..')
sys.path.insert(0, '../src')

tests_path = os.path.dirname(os.path.realpath(__file__))
VQA_Path = os.path.abspath(os.path.join(tests_path, os.pardir))
data_path = os.path.join(VQA_Path, "data")
tests_data_path = os.path.join(data_path, "tests")

def get_test_image_path(file_name):
    return os.path.join(tests_data_path, file_name)
