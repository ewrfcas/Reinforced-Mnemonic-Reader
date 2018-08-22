import tensorflow as tf
from layers import total_params, align_block, summary_vector, start_logits, end_logits
import tensorflow_hub as hub
from tensorflow.contrib.keras import layers
import numpy as np

class Model(object):
    def __init__(self, config, word_mat=None, char_mat=None, test=False, elmo_path=None):

        # hyper-parameter
        self.char_dim = config['char_dim']
        self.cont_limit = config['cont_limit'] if not test else 1000
        self.ques_limit = config['ques_limit'] if not test else 50
        self.char_limit = config['char_limit']
        self.ans_limit = config['ans_limit']
        self.filters = config['filters']
        self.batch_size = config['batch_size']
        self.l2_norm = config['l2_norm']
        self.decay = config['decay']
        self.learning_rate = config['learning_rate']
        self.grad_clip = config['grad_clip']
        self.init_lambda = config['init_lambda']
        self.elmo_path = elmo_path
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")

        # embedding layer
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32), trainable=False)
        self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(char_mat, dtype=tf.float32), trainable=True)

        # input tensor
        self.contw_input_ = tf.placeholder(tf.int32, [None, self.cont_limit], "context_word")
        self.quesw_input_ = tf.placeholder(tf.int32, [None, self.ques_limit], "question_word")
        self.contc_input_ = tf.placeholder(tf.int32, [None, self.cont_limit, self.char_limit], "context_char")
        self.quesc_input_ = tf.placeholder(tf.int32, [None, self.ques_limit, self.char_limit], "question_char")
        self.y_start_ = tf.placeholder(tf.int32, [None, self.cont_limit], "answer_start_index")
        self.y_end_ = tf.placeholder(tf.int32, [None, self.cont_limit], "answer_end_index")
        self.contw_strings = tf.placeholder(tf.string, [None, self.cont_limit], 'contw_strings')
        self.quesw_strings = tf.placeholder(tf.string, [None, self.ques_limit], 'quesw_strings')

        # get mask & length for words & chars
        self.c_mask = tf.cast(self.contw_input_, tf.bool)
        self.q_mask = tf.cast(self.quesw_input_, tf.bool)
        self.ch_mask = tf.cast(self.contc_input_, tf.bool)
        self.qh_mask = tf.cast(self.quesc_input_, tf.bool)
        self.cont_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.ques_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        # elmo features
        if self.elmo_path is not None:
            elmo = hub.Module(self.elmo_path, trainable=True)
            self.cont_elmo = \
            elmo(inputs={"tokens": self.contw_strings, "sequence_len": self.cont_len}, signature="tokens",
                 as_dict=True)["elmo"]
            self.ques_elmo = \
            elmo(inputs={"tokens": self.quesw_strings, "sequence_len": self.ques_len}, signature="tokens",
                 as_dict=True)["elmo"]

        # slice for maxlen in each batch
        self.c_maxlen = tf.reduce_max(self.cont_len)
        self.q_maxlen = tf.reduce_max(self.ques_len)

        self.contw_input = tf.slice(self.contw_input_, [0, 0], [-1, self.c_maxlen])
        self.quesw_input = tf.slice(self.quesw_input_, [0, 0], [-1, self.q_maxlen])
        self.c_mask = tf.slice(self.c_mask, [0, 0], [-1, self.c_maxlen])
        self.q_mask = tf.slice(self.q_mask, [0, 0], [-1, self.q_maxlen])
        self.contc_input = tf.slice(self.contc_input_, [0, 0, 0], [-1, self.c_maxlen, self.char_limit])
        self.quesc_input = tf.slice(self.quesc_input_, [0, 0, 0], [-1, self.q_maxlen, self.char_limit])
        self.y_start = tf.slice(self.y_start_, [0, 0], [-1, self.c_maxlen])
        self.y_end = tf.slice(self.y_end_, [0, 0], [-1, self.c_maxlen])
        if self.elmo_path is not None:
            self.cont_elmo = tf.slice(self.cont_elmo, [0, 0, 0], [-1, self.c_maxlen, 1024])
            self.ques_elmo = tf.slice(self.ques_elmo, [0, 0, 0], [-1, self.q_maxlen, 1024])

        # initial model & complie
        self.build_model()
        total_params()
        self.complie()

    def build_model(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            with tf.variable_scope("Char_Embedding_Layer"):
                # char embedding
                ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.contc_input), [-1, self.char_limit, self.char_dim])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.quesc_input), [-1, self.char_limit, self.char_dim])
                ch_mask = tf.reshape(self.ch_mask, [-1, self.char_limit])
                qh_mask = tf.reshape(self.qh_mask, [-1, self.char_limit])

                char_bilstm = layers.Bidirectional(layers.LSTM(self.char_dim, name='char_bilstm'))
                ch_emb = char_bilstm(ch_emb) # TODO: mask=ch_mask
                qh_emb = char_bilstm(qh_emb) # TODO: mask=qh_mask
                ch_emb = tf.reshape(ch_emb, [-1, self.c_maxlen, ch_emb.shape[-1]])
                qh_emb = tf.reshape(qh_emb, [-1, self.q_maxlen, ch_emb.shape[-1]])

            with tf.variable_scope("Word_Embedding_Layer"):
                # word embedding
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.contw_input)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.quesw_input)

            c_emb = tf.concat([c_emb, ch_emb], axis=-1)
            q_emb = tf.concat([q_emb, qh_emb], axis=-1)
            if self.elmo_path is not None:
                c_emb = tf.concat([c_emb, self.cont_elmo], axis=-1)
                q_emb = tf.concat([q_emb, self.ques_elmo], axis=-1)
            c_emb = tf.nn.dropout(c_emb, 1.0 - self.dropout)
            q_emb = tf.nn.dropout(q_emb, 1.0 - self.dropout)

            # BiLSTM Embedding
            inputs_bilstm = layers.Bidirectional(layers.LSTM(self.filters//2, name='inputs_bilstm', return_sequences=True))
            c_emb = tf.nn.dropout(inputs_bilstm(c_emb, mask=self.c_mask), 1.0 - self.dropout)
            q_emb = tf.nn.dropout(inputs_bilstm(q_emb, mask=self.q_mask), 1.0 - self.dropout)

        with tf.variable_scope("Iterative_Reattention_Aligner"):
            self.Lambda = tf.get_variable('Lambda', dtype=tf.float32, initializer=self.init_lambda)
            with tf.variable_scope("Aligning_Block1"):
                R, Z1, E, B = align_block(u=c_emb,
                                          v=q_emb,
                                          c_mask=self.c_mask,
                                          q_mask=self.q_mask,
                                          Lambda=self.Lambda,
                                          filters=self.filters)
                R = tf.nn.dropout(R, 1.0 - self.dropout)
            with tf.variable_scope("Aligning_Block2"):
                R, Z2, E, B = align_block(u=R,
                                          v=q_emb,
                                          c_mask=self.c_mask,
                                          q_mask=self.q_mask,
                                          E_0=E,
                                          B_0=B,
                                          Lambda=self.Lambda,
                                          filters=self.filters)
                R = tf.nn.dropout(R, 1.0 - self.dropout)
            with tf.variable_scope("Aligning_Block3"):
                R, Z3, E, B = align_block(u=R,
                                          v=q_emb,
                                          c_mask=self.c_mask,
                                          q_mask=self.q_mask,
                                          E_0=E,
                                          B_0=B,
                                          Z_0=[Z1, Z2],
                                          Lambda=self.Lambda,
                                          filters=self.filters)
                R = tf.nn.dropout(R, 1.0 - self.dropout)

        with tf.variable_scope("Answer_Pointer"):
            # logits
            s = summary_vector(q_emb, self.c_maxlen, mask=self.q_mask)
            logits1 = start_logits(R, s, mask=self.c_mask, filters=self.filters)
            logits2 = end_logits(R, logits1, s, mask=self.c_mask, filters=self.filters)

            # get loss
            start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1, labels=self.y_start)
            end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=self.y_end)
            self.loss = tf.reduce_mean(start_loss + end_loss)

            # l2 loss
            if self.l2_norm is not None:
                decay_costs = []
                for var in tf.trainable_variables():
                    decay_costs.append(tf.nn.l2_loss(var))
                self.loss += tf.multiply(self.l2_norm, tf.add_n(decay_costs))

            # output
            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, self.ans_limit)
            self.output1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.output2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

        # EMA
        if self.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(self.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)
                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v is not None:
                        self.assign_vars.append(tf.assign(var, v))

    def complie(self):
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)


# config = {
#     'char_dim': 64,
#     'cont_limit': 400,
#     'ques_limit': 50,
#     'char_limit': 16,
#     'ans_limit': 50,
#     'filters': 100,
#     'num_heads': 1,
#     'dropout': 0.1,
#     'l2_norm': 3e-7,
#     'decay': 0.9999,
#     'learning_rate': 8e-4,
#     'grad_clip': 5.0,
#     'batch_size': 32,
#     'epoch': 10,
#     'init_lambda': 3.0,
#     'path': 'RMRV0'
# }
# model=Model(config=config, word_mat=np.random.random((10000,300)), char_mat=np.random.random((1000,64)))