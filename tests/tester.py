import tests_basis
import imp
import sys
import resnet_test
import question_test
import annotations_fetcher_test
import img_fetcher_test
import question_fetcher_test
import model_dimensionality_test
import top5_accuracy_test

def _set_tests_modules():
    tests_basis.add_test_file(resnet_test)
    tests_basis.add_test_file(question_test)
    tests_basis.add_test_file(annotations_fetcher_test)
    tests_basis.add_test_file(img_fetcher_test)
    tests_basis.add_test_file(question_fetcher_test)
    tests_basis.add_test_file(model_dimensionality_test)
    tests_basis.add_test_file(top5_accuracy_test)


def _reload_modules():
    for module in tests_basis._testes_files_list:
            imp.reload(module)


def _basic_run_tests(args):
    tests_basis.set_options(args)
    tests_basis.run_tests()


def run_tests(args=""):
    global first_time
    if first_time:
        _set_tests_modules()
        first_time = False
    else:
        _reload_modules()

    _basic_run_tests(args)


first_time = True

if __name__ == "__main__":
    run_tests(sys.argv)
