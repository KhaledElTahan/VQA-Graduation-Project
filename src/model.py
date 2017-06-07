import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from data_fetching.data_fetcher import DataFetcher
import numpy as np
import os 

_MAIN_MODEL_GRAPH = None

def dense_batch_relu(input_ph, phase, output_size, name=None):
    h1 = tf.contrib.layers.fully_connected(input_ph, output_size)
    h2 = tf.contrib.layers.batch_norm(h1, is_training=phase)
    return tf.nn.relu(h2, name)

# question_ph is batchSize*#wordsInEachQuestion*300
def question_lstm_model(questions_ph, phase_ph, questions_length_ph, cell_size, layers_num):
    
    mcell = rnn.MultiRNNCell([rnn.LSTMCell(cell_size, state_is_tuple=True) for _ in range(layers_num)])

    init_state = mcell.zero_state(tf.shape(questions_ph)[0], tf.float32) 
    _, final_state = tf.nn.dynamic_rnn(mcell, questions_ph, sequence_length=questions_length_ph, initial_state=init_state)
    
    combined_states = tf.stack(final_state, 1)
    combined_states = tf.reshape(combined_states, [-1, cell_size * layers_num * 2])

    return dense_batch_relu(combined_states, phase_ph, 1024)  # The questions features

def abstract_model(questions_ph, img_features_ph, questions_length_ph, phase_ph, cell_size=512, layers_num=2):

    question_features = question_lstm_model(questions_ph, phase_ph, questions_length_ph, cell_size, layers_num)
    img_features = dense_batch_relu(img_features_ph, phase_ph, 1024)

    fused_features_first = tf.multiply(img_features, question_features)
    fused_features_second = dense_batch_relu(fused_features_first, phase_ph, 1000)
    
    return layers.fully_connected(fused_features_second, 1000)  # logits

def _accuracy(predictions, labels):  # Top 1000 accuracy
    
    _, top_indices = tf.nn.top_k(predictions, k=5, sorted=True, name=None)

    x = tf.to_int32(tf.shape(top_indices))[0]
    y = tf.to_int32(tf.shape(top_indices))[1]
    flattened_ind = tf.range(0, tf.multiply(x, y)) // y * tf.shape(labels)[1] + tf.reshape(top_indices, [-1])

    acc = tf.reduce_sum(tf.gather(tf.reshape(labels, [-1]), flattened_ind)) / tf.to_float(tf.shape(labels))[0] * 100
    return tf.identity(acc, name='accuarcy')

def save_state(saver, sess, starting_pos, idx, batch_size, loss_sum, accuracy_sum, cnt_iteration, cnt_examples, epoch_number):

    directory = os.path.join(os.getcwd(), "models/VQA_model/")
    if not os.path.exists(directory):
        os.makedirs(directory)

    saver.save(sess, os.path.join(os.getcwd(), "models/VQA_model/main_model"), global_step=starting_pos + idx * batch_size)
    np.savetxt('models/VQA_model/statistics.out', (loss_sum, accuracy_sum, cnt_iteration, cnt_examples, epoch_number))

def validation_acc_loss(sess,
                        batch_size,
                        images_place_holder,
                        questions_place_holder,
                        labels_place_holder,
                        questions_length_place_holder,
                        phase_ph,
                        accuracy,
                        loss):

    print("VALIDATION:: STARTING...")

    temp_acc = 0.0
    temp_loss = 0.0
    
    itr = 0

    val_data_fetcher = DataFetcher('validation', batch_size=batch_size)

    while True:

        images_batch, questions_batch, questions_length, labels_batch, end_of_epoch = val_data_fetcher.get_next_batch() 
        
        feed_dict = {questions_place_holder: questions_batch, images_place_holder: images_batch, labels_place_holder: labels_batch, questions_length_place_holder:questions_length, phase_ph: 0}
        l, a = sess.run([loss, accuracy], feed_dict=feed_dict)
        
        itr += 1
        temp_acc += a
        temp_loss += l

        print("VALIDATION:: Iteration[{}]".format(itr))

        if(end_of_epoch):
            break
    
    temp_acc /= itr
    temp_loss /= itr
    
    print("VALIDATION:: ENDING...")

    return temp_loss, temp_acc 

def train_model(number_of_iteration,
                check_point_iteration,
                validation_per_epoch,
                learning_rate, 
                batch_size,
                from_scratch=False,
                validate=True, 
                trace=False):
                    
    sess = tf.Session()
    
    if from_scratch:
        questions_place_holder, images_place_holder, labels_place_holder, questions_length_place_holder, logits, loss, accuarcy, phase_ph = _train_from_scratch(sess) 
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train_step = optimizer.minimize(loss, name='train_step')

        init = tf.initialize_all_variables()
        sess.run(init)

        starting_pos = 0
        loss_sum, accuracy_sum, cnt_iteration, cnt_examples = 0.0, 0.0, 0.0, 0.0
        epoch_number = 1
    else:
        questions_place_holder, images_place_holder, labels_place_holder, questions_length_place_holder, logits, loss, accuarcy, phase_ph, starting_pos, train_step = _get_saved_graph_tensors(sess)
        loss_sum, accuracy_sum, cnt_iteration, cnt_examples, epoch_number = np.loadtxt('models/VQA_model/statistics.out')

    saver = tf.train.Saver(max_to_keep=1)

    train_data_fetcher = DataFetcher('training', batch_size=batch_size, start_itr=starting_pos)

    for i in range(1, number_of_iteration + 1):

        images_batch, questions_batch, questions_length, labels_batch, end_of_epoch = train_data_fetcher.get_next_batch()

        if end_of_epoch: # what if saved then crashed immediately? then validation is lost # BUG
            # print epoch shit here
            epoch_number = epoch_number + 1
            loss_sum, accuracy_sum, cnt_iteration, cnt_examples = 0.0, 0.0, 0.0, 0.0
            save_state(saver, sess, starting_pos, i, batch_size, loss_sum, accuracy_sum, cnt_iteration, cnt_examples, epoch_number)

        feed_dict = {questions_place_holder: questions_batch,
                     images_place_holder: images_batch, 
                     labels_place_holder: labels_batch, 
                     questions_length_place_holder: questions_length, 
                     phase_ph: 1}
        
        _, training_loss, training_acc = sess.run([train_step, loss, accuarcy], feed_dict=feed_dict)
        
        cnt_iteration += 1
        cnt_examples += batch_size
        loss_sum += training_loss
        accuracy_sum += training_acc

        if validate and end_of_epoch:
            validation_loss, validation_acc = validation_acc_loss(sess,
                                                                  batch_size,
                                                                  images_place_holder,
                                                                  questions_place_holder,
                                                                  labels_place_holder,
                                                                  questions_length_place_holder,
                                                                  phase_ph, accuarcy, loss)
            # print validation shit here

        if i % check_point_iteration == 0 and not end_of_epoch:
            save_state(saver, sess, starting_pos, i, batch_size, loss_sum, accuracy_sum, cnt_iteration, cnt_examples, epoch_number)
        
        if trace:
            # trace is only for training log
            # add some statistics like epoch number
            print('TRAINING:: Iteration[{}]: (Accuracy: {}%, Loss: {})'.format(i, training_acc, training_loss))
        
    sess.close()

def _load_model(sess):
    meta_graph_path, data_path, last_index = _get_last_main_model_path()
    new_saver = tf.train.import_meta_graph(meta_graph_path)

    # requires a session in which the graph was launched.
    new_saver.restore(sess, data_path)
    
    global _MAIN_MODEL_GRAPH
    _MAIN_MODEL_GRAPH = tf.get_default_graph()
    return last_index

def _get_last_main_model_path():
    checkpoint_file = open('./models/VQA_model/checkpoint', 'r')
    meta_graph_path = None
    data_path = None
    lst_indx = 0
    for line in checkpoint_file: 
        final_line = line
    word = None
  
    if final_line is not None:
        strt = final_line.find("main_model", 0)
        if strt != -1:
            word = final_line[strt:len(final_line) - 2]
            strt2 = word.find("-", 0)
            lst_indx = int(word[strt2 + 1:len(word)])
            
    if word is not None:
        meta_graph_path = "./models/VQA_model/" + word + ".meta"
        data_path = "./models/VQA_model/" + word
    return meta_graph_path, data_path, lst_indx

def _train_from_scratch(sess):
    questions_place_holder = tf.placeholder(tf.float32, [None, None, 300], name='questions_place_holder') 
    images_place_holder = tf.placeholder(tf.float32, [None, 2048], name='images_place_holder')
    labels_place_holder = tf.placeholder(tf.float32, [None, 1000], name='labels_place_holder')
    questions_length_place_holder = tf.placeholder(tf.int32, [None], name='questions_length_place_holder')
    
    bn_phase = tf.placeholder(tf.bool, [], name='bn_phase')

    logits = tf.identity(abstract_model(questions_place_holder, images_place_holder, questions_length_place_holder, bn_phase), name="logits")
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_place_holder), name='loss')
    accuarcy = _accuracy(tf.nn.softmax(logits), labels_place_holder)

    return questions_place_holder, images_place_holder, labels_place_holder, questions_length_place_holder, logits, loss, accuarcy, bn_phase

def _get_saved_graph_tensors(sess):
    
    last_index = 0
    if _MAIN_MODEL_GRAPH is None:
        last_index = _load_model(sess)
    
    questions_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("questions_place_holder:0") 
    images_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("images_place_holder:0")
    labels_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("labels_place_holder:0")
    questions_length_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("questions_length_place_holder:0")
    
    bn_phase = _MAIN_MODEL_GRAPH.get_tensor_by_name("bn_phase:0")

    logits = _MAIN_MODEL_GRAPH.get_tensor_by_name("logits:0")
    loss = _MAIN_MODEL_GRAPH.get_tensor_by_name("loss:0")
    accuarcy = _MAIN_MODEL_GRAPH.get_tensor_by_name("accuarcy:0")

    train_step = _MAIN_MODEL_GRAPH.get_operation_by_name("train_step")

    return questions_place_holder, images_place_holder, labels_place_holder, questions_length_place_holder, logits, loss, accuarcy, bn_phase, last_index, train_step

def evaluate(image_features, question_features, questions_length):

    sess = tf.Session()
    questions_place_holder, images_place_holder, labels_place_holder, questions_length_place_holder, logits, _, _, phase_ph = _get_saved_graph_tensors(sess)
    feed_dict = {questions_place_holder: question_features, images_place_holder: image_features, questions_length_place_holder: questions_length, phase_ph: 0}
    
    results = tf.nn.softmax(logits)
    evaluation_logits = sess.run([results], feed_dict=feed_dict)

    return evaluation_logits
