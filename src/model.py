import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

def question_lstm_model(questions_ph, cell_size, layers_num):
    cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=False)
    mcell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers_num)
    init_state = mcell.zero_state(batch_size, tf.float32) 
    _, final_state = tf.nn.dynamic_rnn(mcell, questions_ph, initial_state=init_state)

    combined_states = tf.stack([final_state[0],final_state[1]],1)
    combined_states = tf.reshape(combined_states , [-1,cell_size * layers_num * 2])

    return layers.relu(combined_states,1024)  # The questions features

def abstrsct_model(questions_ph, img_features_ph, batch_size=32, cell_size=512, layers_num=2):

    question_features = question_lstm_model(questions_ph, cell_size, layers_num)
    img_features = layers.relu(img_features_ph,1024)

    fused_features_first = tf.multiply(img_features,question_features)
    fused_features_second = layers.relu(fused_features_first,1000)
    
    return layers.linear(fused_features_second,1000)  # logits
