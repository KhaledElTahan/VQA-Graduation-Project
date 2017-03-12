import tests_basis
import sys
from data_fetching.img_fetcher import get_img_batch


def test_fn(args):
    batch = get_img_batch(args[0], args[1], args[2])
    return batch.shape


def main(starting_counter):
    test_args, test_exps = [], []

    test_args.append([29994, 6, False])
    test_exps.append((6, 224, 224, 4))

    test_args.append([29900, 32, False])
    test_exps.append((32, 224, 224, 4))

    test_args.append([20000, 20, False])
    test_exps.append((20, 224, 224, 4))

    tests_basis.create_tests([test_fn] * len(test_args), test_args, test_exps)
    return tests_basis.main_tester("Testing image batch loading", starting_counter)

if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
