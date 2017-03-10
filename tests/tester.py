import tests_basis
import sys
import resnet_test
import question_test
import annotations_fetcher_test
import img_fetcher_test
import question_fetcher_test
import model_dimensionality_test

def main(cmd_variables):
    tests_basis.add_test_file(resnet_test)
    tests_basis.add_test_file(question_test)
    tests_basis.add_test_file(annotations_fetcher_test)
    tests_basis.add_test_file(img_fetcher_test)
    tests_basis.add_test_file(question_fetcher_test)
    tests_basis.add_test_file(model_dimensionality_test)

    tests_basis.set_options(cmd_variables)
    tests_basis.run_tests()

if __name__ == "__main__":
    main(sys.argv)
