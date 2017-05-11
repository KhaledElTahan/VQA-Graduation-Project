import tensorflow as tf
from tensorflow.contrib import layers

_MAIN_MODEL_GRAPH = None

def dense_batch_relu(input_ph, phase, output_size, name=None):
    h1 = tf.contrib.layers.fully_connected(input_ph, output_size)
    h2 = tf.contrib.layers.batch_norm(h1, is_training=phase)
    return tf.nn.relu(h2, name)

#question_ph is batchSize*#wordsInEachQuestion*300
def question_lstm_model(questions_ph, phase_ph, cell_size, layers_num):
    
    cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
    mcell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers_num)
    init_state = mcell.zero_state(tf.shape(questions_ph)[0], tf.float32) 
    _, final_state = tf.nn.dynamic_rnn(mcell, questions_ph, initial_state=init_state)
    
    combined_states = tf.stack(final_state, 1)
    combined_states = tf.reshape(combined_states, [-1, cell_size * layers_num * 2])

    return dense_batch_relu(combined_states, phase_ph, 1024)  # The questions features

def abstract_model(questions_ph, img_features_ph, phase_ph, cell_size=512, layers_num=2):

    question_features = question_lstm_model(questions_ph, phase_ph, cell_size, layers_num)
    img_features = dense_batch_relu(img_features_ph, phase_ph, 1024)

    fused_features_first = tf.multiply(img_features, question_features)
    fused_features_second = dense_batch_relu(fused_features_first, phase_ph, 1000)
    
    return layers.fully_connected(fused_features_second, 1000)  # logits

def _accuracy(predictions, labels):  # Top 1000 accuracy
    
    _, top_indices = tf.nn.top_k(predictions, k=5, sorted=True, name=None)

    x = tf.to_int32(tf.shape(top_indices))[0]
    y = tf.to_int32(tf.shape(top_indices))[1]
    flattened_ind = tf.range(0, tf.mul(x, y)) // y * tf.shape(labels)[1] + tf.reshape(top_indices, [-1])

    acc = tf.reduce_sum(tf.gather(tf.reshape(labels, [-1]), flattened_ind)) / tf.to_float(tf.shape(labels))[0] * 100
    return tf.identity(acc, name='accuarcy')

def validation_acc_loss(sess,
                        batch_size,
                        images_place_holder,
                        questions_place_holder,
                        labels_place_holder,
                        phase_ph,
                        get_data_batch_f,
                        accuracy,
                        loss):
    temp_acc = 0.0
    temp_loss = 0.0
    
    itr = 0
    while True:
        images_batch, questions_batch, labels_batch, end_of_data = get_data_batch_f(itr * batch_size, batch_size, training_data=False)
        if(end_of_data):
            break
        
        feed_dict = {questions_place_holder: questions_batch, images_place_holder: images_batch, labels_place_holder: labels_batch, phase_ph: 0}
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
                get_data_batch_f,
                batch_size,
                from_scratch=False,
                validate=True,
                trace=False):
                    
    sess = tf.Session()
    
    if from_scratch:
        questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy, phase_ph = _train_from_scratch(sess) 
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy, phase_ph, starting_pos = _get_saved_graph_tensors(sess)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before performing the train_step
        train_step = optimizer.minimize(loss)

    saver = tf.train.Saver(max_to_keep=5)
    
    for i in range(number_of_iteration):
        images_batch, questions_batch, labels_batch,_  = get_data_batch_f(starting_pos + i * batch_size, batch_size, training_data=True)
        feed_dict = {questions_place_holder: questions_batch, images_place_holder: images_batch, labels_place_holder: labels_batch, phase_ph: 1}
        
        _, training_loss, training_acc = sess.run([train_step, loss, accuarcy], feed_dict=feed_dict)
        
        if validate and i and i % validation_point_iteration == 0:
            validation_loss, validation_acc = validation_acc_loss(sess,
                                                                  images_place_holder,
                                                                  questions_place_holder,
                                                                  labels_place_holder,
                                                                  phase_ph,
                                                                  get_data_batch_f, accuarcy, loss)
        
        if i and i % check_point_iteration == 0:
            saver.save(sess, os.path.join(os.getcwd(), "main_model"), global_step=starting_pos + (i + 1) * batch_size)
        
        # if trace:
            # _print_statistics()
            # print("Training Loss :", training_loss_result)
            # print("Training Accuracy :", training_acc_result)
        
    sess.close()

def _load_model(sess):
    meta_graph_path, data_path , last_index = _get_last_main_model_path()
    new_saver = tf.train.import_meta_graph(meta_graph_path)

    # requires a session in which the graph was launched.
    new_saver.restore(sess, data_path)
    
    global _MAIN_MODEL_GRAPH
    _MAIN_MODEL_GRAPH = tf.get_default_graph()
    return last_index

def _get_last_main_model_path():
    path = "model_data/"

    checkpoint_file = open('checkpoint','r')
    meta_graph_path = None
    data_path = None
    lst_indx = 0
    for line in checkpoint_file: 
        final_line = line
    word = None
  
    if final_line != None:
        strt = final_line.find("main_model",0)
        if strt != -1:
            word = final_line[strt:len(final_line)-2]
            strt2 = word.find("-",0)
            lst_indx = int(word[strt2+1:len(word)])
            
    if word != None :
        meta_graph_path = word +".meta"
        data_path = "./" + word
    return meta_graph_path, data_path , lst_indx

def _train_from_scratch(sess):
    questions_place_holder = tf.placeholder(tf.float32, [None, None, 300], name='questions_place_holder') 
    images_place_holder = tf.placeholder(tf.float32, [None, 2048], name='imagess_place_holder')
    labels_place_holder = tf.placeholder(tf.float32, [None, 1000], name='labels_place_holder')
    
    bn_phase = tf.placeholder(tf.bool, [], name='bn_phase')

    logits = tf.identity(abstract_model(questions_place_holder, images_place_holder, bn_phase), name="logits")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels_place_holder), name='loss')
    accuarcy = _accuracy(tf.nn.softmax(logits), labels_place_holder) 

    return questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy, bn_phase

def _get_saved_graph_tensors(sess):
    
    last_index = 0
    if _MAIN_MODEL_GRAPH is None:
        last_index = _load_model(sess)
    
    questions_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("questions_place_holder:0") 
    images_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("images_place_holder:0")
    labels_place_holder = _MAIN_MODEL_GRAPH.get_tensor_by_name("labels_place_holder:0")
    
    bn_phase = _MAIN_MODEL_GRAPH.get_tensor_by_name("bn_phase:0")

    logits = _MAIN_MODEL_GRAPH.get_tensor_by_name("logits:0")
    loss = _MAIN_MODEL_GRAPH.get_tensor_by_name("loss:0")
    accuarcy = _MAIN_MODEL_GRAPH.get_tensor_by_name("accuarcy:0")

    return questions_place_holder, images_place_holder, labels_place_holder, logits, loss, accuarcy, bn_phase, last_index

def evaluate(image_features, question_features):

    sess = tf.Session()
    questions_place_holder, images_place_holder, labels_place_holder, logits, _, _, phase_ph = _get_saved_graph_tensors(sess)

    feed_dict = {questions_place_holder: question_features, images_place_holder: image_features, phase_ph: 0}
    
    results = tf.nn.softmax(logits)

    evaluation_logits = sess.run([results], feed_dict=feed_dict)

