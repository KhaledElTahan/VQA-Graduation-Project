import sys
import os

sys.path.insert(0, '../..')
sys.path.insert(0, '../src')

tests_path = os.path.dirname(os.path.realpath(__file__))
VQA_Path = os.path.abspath(os.path.join(tests_path, os.pardir))
data_path = os.path.join(VQA_Path, "data")
tests_data_path = os.path.join(data_path, "tests")

TEST_SUCCESS = 0
TEST_ERROR = 1
TEST_FAIL = 2

def get_test_image_path(file_name):
    return os.path.join(tests_data_path, file_name)

def test(test_fn, args=None, expected_output=None):
    try:
        if isinstance(args, tuple):
            actual_output = test_fn(*args)
        elif args is not None:
            actual_output = test_fn(args)
        else:
            actual_output = test_fn()

        if expected_output is None and actual_output is None:
            return TEST_SUCCESS, None, None
        elif expected_output == actual_output:
            return TEST_SUCCESS, None, None
        else:
            return TEST_ERROR, actual_output, expected_output

    except Exception as e:
        return TEST_FAIL, str(e), None

def main_tester(test_name, starting_count, tests_results):
    line_len = 100

    print('*' * line_len)
    header_1 = '* Testing: {}'.format(test_name)
    space_len_1 = line_len - len(header_1) - 1
    header_2 = '* Number of tests: {}'.format(len(tests_results))
    space_len_2 = line_len - len(header_2) - 1
    print(header_1, ' ' * space_len_1, '*', sep='')
    print(header_2, ' ' * space_len_2, '*', sep='')
    print('*' * line_len)

    success, error, fail = 0, 0, 0

    for t in tests_results:
        test_num = 'Test({})'.format(starting_count)

        if t[0] == TEST_SUCCESS:
            dot_len = line_len - len(test_num) - 6
            print(test_num, '.' * dot_len, "[PASS]", sep='')
            success = success + 1 
        elif t[0] == TEST_ERROR:
            dot_len = line_len - len(test_num) - 7
            print(test_num, '.' * dot_len, "[ERROR]", sep='')
            print(" " * (len(test_num) - 1), "Excpected:", t[2])
            print(" " * (len(test_num) - 1), "Found:", t[1])
            error = error + 1
        else: 
            dot_len = line_len - len(test_num) - 6
            print(test_num, '.' * dot_len, "[FAIL]", sep='')
            print(t[1])
            fail = fail + 1

        starting_count = starting_count + 1

    print()
    print("PASS:  ", success, "/", len(tests_results), sep='', )
    print("ERROR: ", error, "/", len(tests_results), sep='')
    print("FAIL:  ", fail, "/", len(tests_results), sep='')
    print()

    return len(tests_results), success, error, fail