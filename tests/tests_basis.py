import sys
import os
import traceback
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, '../..')
sys.path.insert(0, '../src')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tests_path = os.path.dirname(os.path.realpath(__file__))
VQA_Path = os.path.abspath(os.path.join(tests_path, os.pardir))
data_path = os.path.join(VQA_Path, "data")
tests_data_path = os.path.join(data_path, "tests")

TEST_SUCCESS = 0
TEST_ERROR = 1
TEST_FAIL = 2

_FULL_TRACE = False

_testes_files_list = []

def set_options(cmd_variables):
    global _FULL_TRACE

    for arg in cmd_variables:
        if arg == "-f":
            _FULL_TRACE = True
        elif arg == "-tf":
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        elif arg == "-w":
            warnings.filterwarnings("default")
            
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
        err = None

        if _FULL_TRACE:
            err = traceback.format_exc()
        else:
            err = str(e)

        return TEST_FAIL, err, None

_GLOBAL_LINE_LEN = 100

def main_tester(test_name, starting_count, tests_results):
    line_len = _GLOBAL_LINE_LEN

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

def add_test_file(test_file):
    _testes_files_list.append(test_file)


def _print_row(a, b, c, d, e, len_a, len_b=15, len_c=15, len_d=15, len_e=15, first=False):
    if first:
        print('+', '-' * (len_a - 2), '+', '-' * (len_b - 1), '+', '-' * (len_c - 1), '+', '-' * (len_d - 1), '+', '-' * (len_e - 1), '+', sep='')

    print('|', a, ' ' * (len_a - len(a) - 2), '|', sep='', end='')
    print(b, ' ' * (len_b - len(b) - 1), '|', sep='', end='')
    print(c, ' ' * (len_c - len(c) - 1), '|', sep='', end='')
    print(d, ' ' * (len_d - len(d) - 1), '|', sep='', end='')
    print(e, ' ' * (len_e - len(e) - 1), '|', sep='')

    print('+', '-' * (len_a - 2), '+', '-' * (len_b - 1), '+', '-' * (len_c - 1), '+', '-' * (len_d - 1), '+', '-' * (len_e - 1), '+', sep='')


def run_tests():
    line_len = _GLOBAL_LINE_LEN
    n_sum, s_sum, e_sum, f_sum = 0, 0, 0, 0
    nn, ss, ee, ff = [], [], [], []

    for test_file in _testes_files_list:
        n, s, e, f = test_file.main(n_sum + 1)

        nn.append('{}'.format(n))
        ss.append('{}'.format(s))
        ee.append('{}'.format(e))
        ff.append('{}'.format(f))

        n_sum += n
        s_sum += s
        e_sum += e
        f_sum += f

    _print_row("Test File", "#Tests", "Pass", "Error", "Fail", line_len - 60, first=True)

    i = 0
    for test_file in _testes_files_list:
        _print_row(test_file.__name__, nn[i], ss[i], ee[i], ff[i], line_len - 60)
        i = i + 1

    if n_sum == s_sum:
        print("All Tests passed successfully !")

