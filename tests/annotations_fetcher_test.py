import tests_basis
import sys
from data_fetching.annotation_fetcher import get_annotation_batch


def test_fn(args):
    batch = get_annotation_batch(args[0], args[1], args[2])
    return batch.shape


def main(starting_counter):
    test_args, test_exps, test_results = [], [], []

    test_args.append([29994, 6, False])
    test_exps.append((6, 3, 10))

    test_args.append([29900, 32, False])
    test_exps.append((32, 3, 10))

    test_args.append([20000, 20, False])
    test_exps.append((20, 3, 10))

    for i in range(len(test_args)):
        test_results.append(tests_basis.test(test_fn, test_args[i], test_exps[i]))

    return tests_basis.main_tester("Testing annotations batch loading", starting_counter, test_results)

if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
