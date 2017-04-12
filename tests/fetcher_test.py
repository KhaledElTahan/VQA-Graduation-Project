import tests_basis
import sys
from data_fetching.fetcher import get_data_batch

def test_fn(args):
    images, questions, annotations = get_data_batch(args[0], args[1], args[2])
    return [images.shape, questions.shape, annotations.shape]


def main(starting_counter):
    test_args, test_exps = [], []

    test_args.append([10002, 100, False])
    test_exps.append([(100, 224, 224, 4), (100, 3), (100, 3, 1000)])

    test_args.append([0, 32, False])
    test_exps.append([(32, 224, 224, 4), (32, 3), (32, 3, 1000)])
    
    test_args.append([9999, 5, False])
    test_exps.append([(5, 224, 224, 4), (5, 3), (5, 3, 1000)])

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing image batch loading", starting_counter)


if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)