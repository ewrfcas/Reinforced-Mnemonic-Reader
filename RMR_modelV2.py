import tensorflow as tf
from layers import total_params, align_block, summary_vector, start_logits, end_logits, BiLSTM, ElmoAttention, \
    ElmoCombineLayer, CoveCombineLayer
from bilm import BidirectionalLanguageModel, all_layers
from keras.models import load_model
from loss import rl_loss
import numpy as np


class Model(object):
    def __init__(self, config, word_mat=None, char_mat_trainable=None, char_mat_fix=None, test=False):

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
        self.gamma_b = config['gamma_b']
        self.gamma_c = config['gamma_c']
        self.use_elmo = config['use_elmo']
        self.use_cove = config['use_cove']
        self.use_feat = config['use_feat']
        self.use_rlloss = config['use_rlloss']
        self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
        self.dropout_rnn = tf.placeholder_with_default(0.0, (), name="dropout_rnn")
        self.dropout_emb = tf.placeholder_with_default(0.0, (), name="dropout_emb")
        self.dropout_att = tf.placeholder_with_default(0.0, (), name="dropout_att")
        self.un_size = tf.placeholder_with_default(self.batch_size, (), name="un_size")
        self.rlw = tf.placeholder_with_default(0.0, (), name="rlloss_weights")

        # embedding layer
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(word_mat, dtype=tf.float32),
                                        trainable=False)
        with tf.variable_scope("Input_Embedding_Mat"):
            self.char_mat = tf.get_variable("char_mat",
                                            initializer=np.concatenate([char_mat_trainable, char_mat_fix], axis=0),
                                            trainable=True)

        # input tensor
        self.contw_input = tf.placeholder(tf.int32, [None, None], "context_word")
        self.quesw_input = tf.placeholder(tf.int32, [None, None], "question_word")
        self.contc_input = tf.placeholder(tf.int32, [None, None, self.char_limit], "context_char")
        self.quesc_input = tf.placeholder(tf.int32, [None, None, self.char_limit], "question_char")
        self.y_start = tf.placeholder(tf.int32, [None, None], "answer_start_index")
        self.y_end = tf.placeholder(tf.int32, [None, None], "answer_end_index")
        self.yp_start = tf.placeholder(tf.int32, [None, None], "plausible_answer_start_index")
        self.yp_end = tf.placeholder(tf.int32, [None, None], "plausible_answer_end_index")
        self.contw_elmo_id = tf.placeholder(tf.int32, [None, None, 50], 'contw_elmo_id')
        self.quesw_elmo_id = tf.placeholder(tf.int32, [None, None, 50], 'quesw_elmo_id')
        if self.use_feat:
            self.cont_feat = tf.placeholder(tf.float32, [None, None, 73], "cont_feat")
            self.ques_feat = tf.placeholder(tf.float32, [None, None, 73], "ques_feat")
        self.old_char_mat = tf.placeholder(tf.float32, [None, None], "old_char_mat")
        self.assign_char_mat = tf.assign(self.char_mat, self.old_char_mat)

        # get mask & length for words & chars
        self.c_mask = tf.cast(self.contw_input, tf.bool)
        self.q_mask = tf.cast(self.quesw_input, tf.bool)
        self.cont_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.ques_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        # slice for maxlen in each batch
        self.c_maxlen = tf.reduce_max(self.cont_len)
        self.q_maxlen = tf.reduce_max(self.ques_len)

        # elmo features
        if self.use_elmo == 2:
            options_file = config['elmo_options_path']
            weight_file = config['elmo_weights_path']
            bilm = BidirectionalLanguageModel(options_file, weight_file)
            self.elmo_cont = all_layers(bilm(self.contw_elmo_id))  # [bs, 3, len, 1024]
            self.elmo_cont = tf.transpose(self.elmo_cont, [0, 2, 1, 3])  # [bs, len, 3, 1024]
            self.elmo_ques = all_layers(bilm(self.quesw_elmo_id))
            self.elmo_ques = tf.transpose(self.elmo_ques, [0, 2, 1, 3])
        elif self.use_elmo == 1:
            self.elmo_cont = tf.placeholder(tf.float32, [None, None, 3, 1024], 'elmo_cont')
            self.elmo_ques = tf.placeholder(tf.float32, [None, None, 3, 1024], 'elmo_ques')

        if self.use_cove == 2:
            with tf.variable_scope('Cove_Layer'):
                self.cove_model = load_model(config['cove_path'])
        elif self.use_cove == 1:
            self.cove_cont = tf.placeholder(tf.float32, [None, None, 2, 600], 'cove_cont')
            self.cove_ques = tf.placeholder(tf.float32, [None, None, 2, 600], 'cove_ques')

        # lr schedule
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)

        self.learning_rate = tf.placeholder_with_default(config['learning_rate'], (), name="learning_rate")
        self.lr = tf.minimum(self.learning_rate,
                             self.learning_rate / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))

        # initial model & complie
        self.build_model()
        total_params()
        self.complie()

    def build_model(self):
        with tf.variable_scope("Input_Embedding_Layer"):
            with tf.variable_scope("Char_Embedding_Layer"):
                # char embedding
                ch_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.contc_input),
                                    [-1, self.char_limit, self.char_dim])
                qh_emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, self.quesc_input),
                                    [-1, self.char_limit, self.char_dim])
                ch_emb = tf.nn.dropout(ch_emb, 1 - self.dropout_emb)
                qh_emb = tf.nn.dropout(qh_emb, 1 - self.dropout_emb)

                ch_emb, qh_emb = BiLSTM([ch_emb, qh_emb], self.char_dim // 2, dropout=self.dropout_rnn,
                                        name='char_lstm', return_state=True)
                ch_emb = tf.reshape(ch_emb, [-1, self.c_maxlen, self.char_dim])
                qh_emb = tf.reshape(qh_emb, [-1, self.q_maxlen, self.char_dim])

            with tf.variable_scope("Word_Embedding_Layer"):
                # word embedding
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.contw_input)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.quesw_input)
                c_emb = tf.nn.dropout(c_emb, 1.0 - self.dropout_emb)
                q_emb = tf.nn.dropout(q_emb, 1.0 - self.dropout_emb)

            # cove features
            if self.use_cove != 0:
                if self.use_cove == 2:
                    self.cove_cont = tf.stop_gradient(self.cove_model(c_emb))  # [bs, c_len, 2, 600]
                    self.cove_ques = tf.stop_gradient(self.cove_model(q_emb))  # [bs, q_len, 2, 600]
                with tf.variable_scope('Cove_weights', reuse=tf.AUTO_REUSE):
                    cove_context_input = CoveCombineLayer(self.cove_cont, 'input')
                    cove_question_input = CoveCombineLayer(self.cove_ques, 'input')
                c_emb = tf.concat([c_emb, cove_context_input], axis=-1)
                q_emb = tf.concat([q_emb, cove_question_input], axis=-1)

            # elmo features
            if self.use_elmo != 0:
                with tf.variable_scope('ELMo_weights', reuse=tf.AUTO_REUSE):
                    elmo_context_input = ElmoCombineLayer(self.elmo_cont, 'input')
                    elmo_question_input = ElmoCombineLayer(self.elmo_ques, 'input')
                    elmo_context_output = ElmoCombineLayer(self.elmo_cont, 'output')
                    elmo_question_output = ElmoCombineLayer(self.elmo_ques, 'output')
                c_emb = tf.concat([c_emb, elmo_context_input], axis=-1)
                q_emb = tf.concat([q_emb, elmo_question_input], axis=-1)

            if self.use_feat:
                c_emb = tf.concat([c_emb, self.cont_feat], axis=-1)
                q_emb = tf.concat([q_emb, self.ques_feat], axis=-1)

            # combine embedding feats
            c_emb = tf.concat([c_emb, ch_emb], axis=-1)
            q_emb = tf.concat([q_emb, qh_emb], axis=-1)

            # BiLSTM Embedding
            with tf.variable_scope("BiLSTM_Embedding_Layer"):
                c_emb, q_emb = BiLSTM([c_emb, q_emb], self.filters // 2, dropout=self.dropout_rnn, name='encoder')

        with tf.variable_scope("Iterative_Reattention_Aligner"):
            self.Lambda = tf.get_variable('Lambda', dtype=tf.float32, initializer=self.init_lambda)
            with tf.variable_scope("Aligning_Block1"):
                R, Z1, E, B = align_block(u=c_emb,
                                          v=q_emb,
                                          c_mask=self.c_mask,
                                          q_mask=self.q_mask,
                                          Lambda=self.Lambda,
                                          filters=self.filters,
                                          dropout=self.dropout_rnn)
                R = tf.nn.dropout(R, 1.0 - self.dropout_att)
            with tf.variable_scope("Aligning_Block2"):
                R, Z2, E, B = align_block(u=R,
                                          v=q_emb,
                                          c_mask=self.c_mask,
                                          q_mask=self.q_mask,
                                          E_0=E,
                                          B_0=B,
                                          Lambda=self.Lambda,
                                          filters=self.filters,
                                          dropout=self.dropout_rnn)
                R = tf.nn.dropout(R, 1.0 - self.dropout_att)
            with tf.variable_scope("Aligning_Block3"):
                R, Z3, E, B = align_block(u=R,
                                          v=q_emb,
                                          c_mask=self.c_mask,
                                          q_mask=self.q_mask,
                                          E_0=E,
                                          B_0=B,
                                          Z_0=[Z1, Z2],
                                          Lambda=self.Lambda,
                                          filters=self.filters,
                                          dropout=self.dropout_rnn)
                R = tf.nn.dropout(R, 1.0 - self.dropout_att)

        with tf.variable_scope("Answer_Pointer"):
            # logits
            if self.use_elmo != 0:
                elmo_output_feats = ElmoAttention([elmo_context_output, elmo_question_output],
                                                  self.c_maxlen, self.q_maxlen, self.q_mask, self.dropout)
                R = tf.concat([R, elmo_output_feats], axis=-1)
            s = summary_vector(q_emb, self.c_maxlen, mask=self.q_mask)
            s = tf.nn.dropout(s, 1 - self.dropout)
            logits1 = start_logits(R, s, mask=self.c_mask, filters=self.filters, name='Start_Pointer')  # [bs, c_len]
            logits2 = end_logits(R, logits1, s, mask=self.c_mask, filters=self.filters,
                                 name='End_Pointer')  # [bs, c_len]
            self.unanswer_bias = tf.get_variable("unanswer_bias", [1], initializer=tf.zeros_initializer())
            self.unanswer_bias = tf.reshape(tf.tile(self.unanswer_bias, [self.un_size]), [-1, 1])
            logits1 = tf.concat((self.unanswer_bias, logits1), axis=-1)
            logits2 = tf.concat((self.unanswer_bias, logits2), axis=-1)

            logits1p = start_logits(R, s, mask=self.c_mask, filters=self.filters, name='Start_Pointer2')  # [bs, c_len]
            logits2p = end_logits(R, logits1p, s, mask=self.c_mask, filters=self.filters,
                                  name='End_Pointer2')  # [bs, c_len]

        with tf.variable_scope("Loss_Layer"):
            # maximum-likelihood (ML) loss for dataset V2.0
            # loss a
            start_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1, labels=self.y_start)
            end_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=self.y_end)
            self.loss = tf.reduce_mean(start_loss + end_loss)

            # loss b
            pstart_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1p, labels=self.yp_start)
            pend_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2p, labels=self.yp_end)
            self.loss += self.gamma_b * tf.reduce_mean(pstart_loss + pend_loss)

            # loss c
            answer_exist_label = tf.cast(tf.slice(self.y_start, [0, 0], [-1, 1]), tf.float32)
            self.loss += self.gamma_c * tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.unanswer_bias, labels=answer_exist_label))

            # l2 loss
            if self.l2_norm is not None:
                decay_costs = []
                for var in tf.trainable_variables():
                    decay_costs.append(tf.nn.l2_loss(var))
                self.loss += tf.multiply(self.l2_norm, tf.add_n(decay_costs))

            # RL loss
            if self.use_rlloss:
                with tf.variable_scope("Reinforcement_Loss"):
                    self.rl_loss_a, _, _ = rl_loss(logits1, logits2, self.y_start, self.y_end, self.c_maxlen + 1)
                    self.rl_loss_b, _, _ = rl_loss(logits1p, logits2p, self.yp_start, self.yp_end, self.c_maxlen)
                    self.loss += (self.rlw * (self.rl_loss_a + self.gamma_b * self.rl_loss_b))

        with tf.variable_scope('Output_Layer'):
            softmax_start_scores = tf.nn.softmax(tf.slice(logits1, [0, 1], [-1, -1]))
            softmax_end_scores = tf.nn.softmax(tf.slice(logits2, [0, 1], [-1, -1]))

            unanswer_mask1 = tf.cast(tf.argmax(tf.nn.softmax(logits1), axis=-1), tf.int64)
            unanswer_mask1 = tf.cast(tf.cast(unanswer_mask1, tf.bool), tf.int64)  # [bs,] has answer=1 no answer=0
            unanswer_move1 = unanswer_mask1 - 1  # [bs,] has answer=0 no answer=-1
            unanswer_mask2 = tf.cast(tf.argmax(tf.nn.softmax(logits2), axis=-1), tf.int64)
            unanswer_mask2 = tf.cast(tf.cast(unanswer_mask2, tf.bool), tf.int64)  # [bs,]
            unanswer_move2 = unanswer_mask2 - 1

            softmax_start_p = tf.nn.softmax(logits2p)
            softmax_end_p = tf.nn.softmax(logits2p)
            softmax_start_scores = (1 - self.gamma_b) * softmax_start_scores + self.gamma_b * softmax_start_p
            softmax_end_scores = (1 - self.gamma_b) * softmax_end_scores + self.gamma_b * softmax_end_p

            outer = tf.matmul(tf.expand_dims(softmax_start_scores, axis=2),
                              tf.expand_dims(softmax_end_scores, axis=1))
            outer = tf.matrix_band_part(outer, 0, self.ans_limit)

            def position_encoding(x):
                import math
                for i in range(x.shape[0]):
                    for j in range(x.shape[1]):
                        if j - i > 5:
                            x[i][j] = float(1.0 / math.log(j - i + 1))
                return x

            mask_mat = tf.ones((self.c_maxlen, self.c_maxlen))
            mask_mat = tf.expand_dims(tf.py_func(position_encoding, [mask_mat], tf.float32), axis=0)
            mask_mat = tf.tile(mask_mat, [self.un_size, 1, 1])

            outer_masked = outer * mask_mat
            self.mask_output1 = tf.argmax(tf.reduce_max(outer_masked, axis=2),
                                          axis=1) * unanswer_mask1 + unanswer_move1
            self.mask_output2 = tf.argmax(tf.reduce_max(outer_masked, axis=1),
                                          axis=1) * unanswer_mask2 + unanswer_move2

    def complie(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.8, beta2=0.999, epsilon=1e-7)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables), global_step=self.global_step)

        # EMA
        with tf.variable_scope("EMA_Weights"):
            if self.decay is not None and self.decay < 1.:
                self.var_ema = tf.train.ExponentialMovingAverage(self.decay)
                with tf.control_dependencies([self.train_op]):
                    self.ema_train_op = self.var_ema.apply(
                        list(set(tf.trainable_variables()) ^ set(tf.trainable_variables('Cove_Layer'))))
                # assign ema weights
                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v is not None:
                        self.assign_vars.append(tf.assign(var, v))


# import numpy as np
#
# config = {
#     'char_dim': 64,
#     'cont_limit': 400,
#     'ques_limit': 50,
#     'char_limit': 16,
#     'ans_limit': -1,
#     'filters': 256,
#     'dropout': 0.1,
#     'dropout_emb': 0.1,
#     'l2_norm': 3e-7,
#     'decay': 0.9999,
#     'gamma_c': 1.0,
#     'gamma_b': 0.3,
#     'learning_rate': 1e-3,
#     'grad_clip': 5.0,
#     'init_lambda': 3.0,
#     'loss_type': 'use_plausible',
#     'use_elmo': 0,
#     'use_cove': 0,
#     'use_feat': True,
#     'optimizer': 'adam',
#     'cove_path': 'Keras_CoVe_2layers.h5',
#     'elmo_weights_path': 'elmo_tf/models/squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5',
#     'elmo_options_path': 'elmo_tf/models/options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json',
#     'train_tfrecords': 'tfrecords/train_pre_elmo_cove.tfrecords',
#     'dev_tfrecords': 'tfrecords/dev_pre_elmo_cove.tfrecords',
#     'batch_size': 24,
#     'epoch': 40,
#     'origin_path': None,  # not finetune
#     'path': 'QANetV253'
# }
# word_mat = np.random.random((90950, 300)).astype(np.float32)
# char_mat2 = np.random.random((94, 300)).astype(np.float32)
# char_mat = np.random.random((1171, 300)).astype(np.float32)
# model = Model(config, word_mat, char_mat, char_mat2)
