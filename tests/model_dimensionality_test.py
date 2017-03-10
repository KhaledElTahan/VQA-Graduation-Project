import tests_basis
import sys
import tensorflow as tf
from model import _train_from_scratch

def test_fn(tensor_type):
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy = _train_from_scratch(sess)
    sess.close()

    if tensor_type == "questions_place_holder":
        return questions_place_holder.get_shape().as_list()
    elif tensor_type == "images_place_holder":
        return images_place_holder.get_shape().as_list()
    elif tensor_type == "labels_place_holder":
        return labels_place_holder.get_shape().as_list()
    elif tensor_type == "logits":
        return logits.get_shape().as_list()
    elif tensor_type == "loss":
        return loss.get_shape().as_list()
    return accuarcy.get_shape().as_list()


def main(starting_counter):
    test_args, test_exps, test_results = [], [], []

    test_args.append("questions_place_holder")
    test_exps.append([None, None, 300])
    test_args.append("images_place_holder")
    test_exps.append([None, 2048])
    test_args.append("labels_place_holder")
    test_exps.append([None, 1000])
    test_args.append("logits")
    test_exps.append([None, 1000])
    test_args.append("loss")
    test_exps.append([])
    test_args.append("accuracy")
    test_exps.append([])

    for i in range(len(test_args)):
        with tf.variable_scope('{}'.format(i)):
            test_results.append(tests_basis.test(test_fn, test_args[i], test_exps[i]))

    return tests_basis.main_tester("Testing The model dimentionality based on Batch_Size", starting_counter, test_results)

if __name__ == "__main__":
    tests_basis.set_options(sys.argv)
    main(1)
    