import tests_basis
import resnet_test
import question_test

def main():
    tests_basis.add_test_file(resnet_test)
    tests_basis.add_test_file(question_test)

    tests_basis.run_tests()

if __name__ == "__main__":
    main()
