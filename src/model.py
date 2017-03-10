import tensorflow as tf
from tensorflow.contrib import layers

_MAIN_MODEL_GRAPH = None

def question_lstm_model(questions_ph, cell_size, layers_num, batch_size):
    
    cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=False)
    mcell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers_num)
    init_state = mcell.zero_state(batch_size, tf.float32) 
    _, final_state = tf.nn.dynamic_rnn(mcell, questions_ph, initial_state=init_state)
    
    combined_states = tf.stack(final_state, 1)
    combined_states = tf.reshape(combined_states, [-1, cell_size * layers_num * 2])

    return layers.relu(combined_states, 1024)  # The questions features

def abstract_model(questions_ph, img_features_ph, batch_size, cell_size=512, layers_num=2):

    question_features = question_lstm_model(questions_ph, cell_size, layers_num, batch_size)
    img_features = layers.relu(img_features_ph, 1024)

    fused_features_first = tf.multiply(img_features, question_features)
    fused_features_second = layers.relu(fused_features_first, 1000)
    
    return layers.linear(fused_features_second, 1000)  # logits

def _accuracy(predictions, labels):
    # predictions & labels are [batch_size x 1000]
    # 1 - abs(predic - target)/sum(target)
    
    abs_diff = tf.abs(predictions - labels)
    return (1 - tf.mean(tf.sum(abs_diff, axis=1) / tf.sum(labels, axis=1))) * 100.0

def validation_acc_loss(sess,
                        batch_size,
                        images_place_holder,
                        questions_place_holder,
                        labels_place_holder,
                        get_images_batch_f,
                        get_questions_batch_f,
                        get_labels_batch_f,
                        accuracy,
                        loss):
    temp_acc = 0.0
    temp_loss = 0.0
    
    itr = 0
    while True:
        images_batch, end_of_data = get_images_batch_f(itr * batch_size, batch_size, validation_data=True)
        questions_batch, end_of_data = get_questions_batch_f(itr * batch_size, batch_size, validation_data=True)
        labels_batch, end_of_data = get_labels_batch_f(itr * batch_size, batch_size, validation_data=True)
        
        if(end_of_data):
            break
        
        feed_dict = {questions_place_holder: questions_batch, images_place_holder: images_batch, labels_place_holder: labels_batch}
        l, a = sess.run([loss, accuracy], feed_dict=feed_dict)
        
        itr += 1
        temp_acc += a
        temp_loss += l
    
    temp_acc /= itr
    temp_loss /= itr
    
    return temp_loss, temp_acc 

def train_model(starting_pos,
                number_of_iteration,
                check_point_iteration,
                validation_point_iteration,
                learning_rate, 
                get_images_batch_f,
                get_questions_batch_f,
                get_labels_batch_f,
                batch_size=None,
                from_scratch=False,
                validate=True,
                trace=False):
                    
    sess = tf.Session()
    
    if from_scratch:
        questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy = _train_from_scratch(sess, batch_size) 
    else:
        questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy = _fine_tune_model(sess)
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    train_step = optimizer.minimize(loss)
    
    saver = tf.train.Saver(max_to_keep=5)
    
    # to remove later ## JUST FOR TESTING
    batch_size = 32
    
    for i in range(number_of_iteration):
        images_batch = get_images_batch_f(starting_pos + i * batch_size, batch_size, training_data=True)
        questions_batch = get_questions_batch_f(starting_pos + i * batch_size, batch_size, training_data=True)
        labels_batch = get_labels_batch_f(starting_pos + i * batch_size, batch_size, training_data=True)
        
        feed_dict = {questions_place_holder: questions_batch, images_place_holder: images_batch, labels_place_holder: labels_batch}
        
        _, training_loss, training_acc = sess.run([train_step, loss, accuarcy], feed_dict=feed_dict)
        
        if validate and i and i % validation_point_iteration == 0:
            validation_loss, validation_acc = validation_acc_loss(sess, batch_size,
                                                                  images_place_holder,
                                                                  questions_place_holder,
                                                                  labels_place_holder,
                                                                  get_images_batch_f, get_questions_batch_f,
                                                                  get_labels_batch_f, accuarcy, loss)
        
        if i and i % check_point_iteration == 0:
            saver.save(sess, "main_model_", global_step=starting_pos + (i + 1) * batch_size)
        
        # if trace:
            # _print_statistics()
            # print("Training Loss :", training_loss_result)
            # print("Training Accuracy :", training_acc_result)
        
    sess.close()

def _load_model(sess):
    meta_graph_path, data_path = _get_last_main_model_path()
    new_saver = tf.train.import_meta_graph(meta_graph_path)

    # requires a session in which the graph was launched.
    new_saver.restore(sess, data_path)
    
    global _MAIN_MODEL_GRAPH
    _MAIN_MODEL_GRAPH = tf.get_default_graph()

def _get_last_main_model_path():
    meta_graph_path = None
    data_path = None
    
    return meta_graph_path, data_path

def _train_from_scratch(sess, batch_size):
    questions_place_holder = tf.placeholder(tf.float32, [batch_size, None, 300], name='questions_place_holder') 
    images_place_holder = tf.placeholder(tf.float32, [batch_size, 224, 224, 3], name='imagess_place_holder')
    labels_place_holder = tf.placeholder(tf.float32, [batch_size, 1000], name='labels_place_holder')
    
    logits = tf.identify(abstract_model(questions_place_holder, images_place_holder, batch_size), name="logits")
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels_place_holder), name='loss')
    accuarcy = _accuracy(tf.nn.softmax(logits), labels_place_holder, name='accuarcy') 
    
    return questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy

def _fine_tune_model(sess):
    
    if _MAIN_MODEL_GRAPH is None:
        _load_model(sess)
    
    questions_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("questions_place_holder:0") 
    images_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("images_place_holder:0")
    labels_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("labels_place_holder:0")
    
    logits = _MAIN_MODEL_GRAPH.get_tensor_by_name("logits:0")
    loss = _MAIN_MODEL_GRAPH.get_tensor_by_name("loss:0")
    accuarcy = _MAIN_MODEL_GRAPH.get_tensor_by_name("accuarcy:0")

    return questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy

# TO DO
# Regularization
# Batch normalization
# Evaluation






