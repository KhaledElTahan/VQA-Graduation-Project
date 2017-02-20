import tests_basis
import resnet_test
import question_test

def main():
    main_n, main_s = None, None

    n, s, e, f = resnet_test.main(1)
    main_n = n
    main_s = s
    n, s, e, f = question_test.main(main_n + 1)
    main_n += n
    main_s += s

    if main_s == main_n:
        print("All", main_n, "Tests Passed Successfully !")

if __name__ == "__main__":
    main()
