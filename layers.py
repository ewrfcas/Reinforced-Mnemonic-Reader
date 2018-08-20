import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensor2tensor.layers.common_layers import conv1d
import numpy as np

def exp_mask(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)

def align_block(u, v, c_mask, q_mask, Lambda, filters=128, E_0=None, B_0=None, Z_0=None):
    with tf.variable_scope("Interactive_Alignment"):
        # attention
        u_ = tf.nn.relu(conv1d(u, filters, 1, name="Wu"))
        v_ = tf.nn.relu(conv1d(v, filters, 1, name="Wv"))
        E = tf.matmul(v_, u_, transpose_b=True) # [bs, len_q, len_c]
        if E_0 is not None:
            E += (Lambda * E_0)
        E_ = tf.nn.softmax(exp_mask(E, tf.expand_dims(q_mask, axis=-1)), axis=1) # [bs, len_q, len_c]
        v_E = tf.matmul(E_, v, transpose_a=True) # [bs, len_c, dim]

        # fusion
        uv = tf.concat([u, v_E, u * v_E, u - v_E], axis=-1)
        x = tf.nn.relu(conv1d(uv, filters, 1, name='Wr'))
        g = tf.nn.sigmoid(conv1d(uv, filters, 1, name='Wg'))
        h = g * x + (1 - g) * u # [bs, len_c, dim]

    with tf.variable_scope("Self_Alignment"):
        # attention
        h_1 = tf.nn.relu(conv1d(h, filters, 1, name='Wh1'))
        h_2 = tf.nn.relu(conv1d(h, filters, 1, name='Wh2'))
        B = tf.matmul(h_2, h_1, transpose_b=True) # [bs, len_c, len_c]
        if B_0 is not None:
            B += (Lambda * B_0)
        B_ = tf.nn.softmax(exp_mask(B, tf.expand_dims(c_mask, axis=-1)), axis=1) # [bs, len_c, len_c]
        h_B = tf.matmul(B_, h, transpose_a=True)

        # fusion
        hh = tf.concat([h, h_B, h * h_B, h - h_B], axis=-1)
        x = tf.nn.relu(conv1d(uv, filters, 1, name='Wr'))
        g = tf.nn.sigmoid(conv1d(hh, filters, 1, name='Wg'))
        Z = g * x + (1 - g) * h  # [bs, len_c, dim]

    with tf.variable_scope("Evidence_Collection"):
        if Z_0 is not None:
            Z = tf.concat([Z, Z_0[0], Z_0[1]], axis=-1)
        R = layers.Bidirectional(layers.LSTM(filters//2, return_sequences=True))(Z) # [bs, len_c, dim]

    # return the E_t, B_t
    E_t = tf.nn.softmax(exp_mask(E, tf.expand_dims(c_mask, axis=1)), axis=-1) # [bs, len_q, len_c]
    E_t = tf.matmul(E_t, B_)
    B_t = tf.nn.softmax(exp_mask(B, tf.expand_dims(c_mask, axis=1)), axis=-1) # [bs, len_c, len_c]
    B_t = tf.matmul(B_t, B_)

    return R, Z, E_t, B_t

def summary_vector(q_emb, q_maxlen, mask):
    with tf.variable_scope("Question_Summary"):
        alpha = tf.nn.softmax(exp_mask(tf.squeeze(conv1d(q_emb, 1, 1), axis=-1), mask))
        s = tf.expand_dims(alpha, axis=-1) * q_emb
        s = tf.reduce_sum(s, axis=1, keepdims=True) # [bs, 1, dim]
        s = tf.tile(s, [1, q_maxlen, 1]) # [bs, len_q, dim]
    return s

def start_logits(R, s, mask, filters=128):
    with tf.variable_scope("Start_Pointer"):
        logits1 = tf.concat([R, s, R * s, R - s], axis=-1)
        logits1 = tf.nn.tanh(conv1d(logits1, filters, 1, name='Wt'))
        logits1 = tf.squeeze(conv1d(logits1, 1, 1, name='Wf'), axis=-1)
        logits1 = exp_mask(logits1, mask)
    return logits1

def end_logits(R, logits1, s, mask, filters=128):
    with tf.variable_scope("End_Pointer"):
        l = R * tf.expand_dims(logits1, axis=-1) # [bs, len_c, dim]
        s = tf.reduce_mean(s, axis=1, keepdims=True) # [bs, 1, dim]
        s_ = tf.concat([s, l, s * l, s - l], axis=-1)
        x = tf.nn.relu(conv1d(s_, filters, 1, name='Wr')) # [bs, len_c, dim]
        g = tf.nn.sigmoid(conv1d(s_, filters, 1, name='Wg'))  # [bs, len_c, dim]
        s_ = g * x + (1 - g) * s  # [bs, len_c, dim]

        logits2 = tf.concat([R, s_, R * s_, R - s_], axis=-1)
        logits2 = tf.nn.tanh(conv1d(logits2, filters, 1, name='Wt'))
        logits2 = tf.squeeze(conv1d(logits2, 1, 1, name='Wf'), axis=-1)
        logits2 = exp_mask(logits2, mask)
    return logits2

def total_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total number of trainable parameters: {}".format(total_parameters))

# Lambda = tf.get_variable('Lambda', dtype=tf.float32, initializer=0.3)
# u=tf.constant(np.random.random((300,400,128)),dtype=tf.float32)
# c_mask=tf.constant(np.ones((300,400)))
# v=tf.constant(np.random.random((300,50,128)),dtype=tf.float32)
# q_mask=tf.constant(np.ones((300,50)))
# s=summary_vector(v,q_mask)