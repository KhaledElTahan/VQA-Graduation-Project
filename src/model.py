import tensorflow as tf
import numpy as np

def model(input, batch_size=32, cell_size=512, layers_num=2):
    sess = tf.Session()

    X = tf.placeholder(tf.float32, [batch_size, None, 300]) # Batch of list of 300D Vectors

    cell = tf.nn.rnn_cell.LSTMCell(cell_size, state_is_tuple=True)
    mcell = tf.nn.rnn_cell.MultiRNNCell([cell] * layers_num)
    init_state = mcell.zero_state(batch_size, tf.float32) 
    rnn_outputs, final_state = tf.nn.dynamic_rnn(mcell, X, initial_state=init_state)

    
    
    # init = tf.global_variables_initializer()
    # sess.run(init)
    # sess.run(final_state ,feed_dict={X:input})

def test():
    x = [1] * 300
    y = [1] * 300
    q1 = [x, y]
    q2 = [x, y]
    q3 = [x, y]
    b = [q1, q2, q3]
    batch = np.array(b) # 3 * 2 * 300

    return model(batch, 3, 250, 2)

# test()