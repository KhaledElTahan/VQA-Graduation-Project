import tests_basis
import resnet_test
import question_test
import annotations_fetcher_test
import img_fetcher_test
import question_fetcher_test

def main():
    tests_basis.add_test_file(resnet_test)
    tests_basis.add_test_file(question_test)
    tests_basis.add_test_file(annotations_fetcher_test)
    tests_basis.add_test_file(img_fetcher_test)
    tests_basis.add_test_file(question_fetcher_test)

    tests_basis.run_tests()

if __name__ == "__main__":
    main()
