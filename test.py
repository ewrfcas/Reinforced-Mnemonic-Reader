import os
import numpy as np
import json
import RMR_modelV6_squad2 as RMR
import tensorflow.contrib.slim as slim
from util.util import *
import tensorflow as tf
import pandas as pd
from util.log_wrapper import create_logger

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

if __name__ == '__main__':

    data_source = '../QANet_tf/dataset_pre3'

    config = {
        'char_dim': 300,
        'cont_limit': 400,
        'ques_limit': 50,
        'char_limit': 16,
        'ans_limit': -1,
        'filters': 300,
        'dropout': 0.175,
        'dropout_emb': 0.15,
        'dropout_att': 0.2,
        'dropout_rnn': 0.1,
        'l2_norm': 3e-7,
        'decay': 1,
        'gamma_b': 0.3,
        'gamma_c': 1.0,
        'init_lambda': 3.0,
        'learning_rate': 8e-4,
        'shuffle_size': 25000,
        'grad_clip': 5.0,
        'use_elmo': 0,
        'use_cove': 0,
        'use_feat': True,
        'use_rlloss': True,
        'rlw': 0.0,
        'rlw2': 0.8,
        'optimizer': 'adam',
        'cove_path': '../SAN_tf/Keras_CoVe_2layers.h5',
        'elmo_weights_path': '../SAN_tf/elmo_tf/models/squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5',
        'elmo_options_path': '../SAN_tf/elmo_tf/models/options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json',
        'train_tfrecords': '../QANet_tf/tfrecords/train_pre_elmo_cove3.tfrecords',
        'dev_tfrecords': '../QANet_tf/tfrecords/dev_pre_elmo_cove3.tfrecords',
        'batch_size': 32,
        'epoch': 25,
        'origin_path': None,  # not finetune
        'path': 'RMR005'
    }

    global logger
    logger = create_logger(__name__, to_disk=False)

    logger.info('loading data...')
    dev_qid = np.load(data_source + '/dev_qid.npy').astype(np.int32)
    with open(data_source + '/test_eval.json', "r") as fh:
        eval_file = json.load(fh)

    # load embedding matrix
    logger.info('loading embedding...')
    word_mat = np.load(data_source + '/word_emb_mat.npy')
    char_mat_fix = np.load(data_source + '/char_emb_mat_fix.npy').astype(np.float32)
    char_mat_trainable = np.load(data_source + '/char_emb_mat_trainable.npy').astype(np.float32)


    logger.info('generate dev tfrecords...')
    dev_dataset = tf.data.TFRecordDataset(config['dev_tfrecords']) \
        .map(get_record_parser(config), num_parallel_calls=8) \
        .padded_batch(config['batch_size'], padded_shapes=([None],
                                                           [None],
                                                           [None, None],
                                                           [None, None],
                                                           [None, None],
                                                           [None, None],
                                                           [None, None, None],
                                                           [None, None, None],
                                                           [None, None, None],
                                                           [None, None, None],
                                                           [None],
                                                           [None],
                                                           [None],
                                                           [None]))
    dev_iterator = dev_dataset.make_initializable_iterator()
    dev_next_element = dev_iterator.get_next()
    dev_sum = 11730

    logger.info('init model...')
    model = RMR.Model(config, word_mat=word_mat, char_mat_trainable=char_mat_trainable, char_mat_fix=char_mat_fix)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    best_f1 = 0
    best_em = 0
    f1s = []
    ems = []

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        # scope with trainable weights
        variables_to_restore = slim.get_variables_to_restore(include=['Input_Embedding_Mat',
                                                                      'Input_Embedding_Layer',
                                                                      'Iterative_Reattention_Aligner',
                                                                      'Answer_Pointer',
                                                                      'EMA_Weights'])
        saver = tf.train.Saver(variables_to_restore, max_to_keep=10)
        if config['origin_path'] is not None and os.path.exists(
                os.path.join('model', config['origin_path'], 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(os.path.join('model', str(config['origin_path']) + '/')))

        i_batch = 0
        val_n_batch = dev_sum // config['batch_size'] + 1
        sum_loss = 0

        # validating step
        # save the temp weights and do ema
        if config['decay'] < 1.0:
            sess.run(model.assign_vars)
            print('EMA over...')
        sess.run(dev_iterator.initializer)
        logger.info('validating...')
        sum_loss_val = 0
        y1s = []
        y2s = []
        i_batch = 0
        while True:
            try:
                context_idxs, ques_idxs, \
                context_char_idxs, ques_char_idxs, \
                context_feat, ques_feat, \
                elmo_context_feat, elmo_question_feat, \
                cove_context_feat, cove_question_feat, \
                y1, y2, y1p, y2p = sess.run(dev_next_element)
                feed_dict_ = {model.contw_input: context_idxs, model.quesw_input: ques_idxs,
                              model.contc_input: context_char_idxs, model.quesc_input: ques_char_idxs,
                              model.y_start: y1, model.y_end: y2,
                              model.yp_start: y1p, model.yp_end: y2p,
                              model.un_size: context_idxs.shape[0]}
                if config['use_feat']:
                    feed_dict_[model.cont_feat] = context_feat
                    feed_dict_[model.ques_feat] = ques_feat
                if config['use_elmo'] == 1:
                    feed_dict_[model.elmo_cont] = elmo_context_feat
                    feed_dict_[model.elmo_ques] = elmo_question_feat
                if config['use_cove'] == 1:
                    feed_dict_[model.cove_cont] = cove_context_feat
                    feed_dict_[model.cove_ques] = cove_question_feat
                loss_value, y1, y2 = sess.run([model.loss, model.mask_output1, model.mask_output2],
                                              feed_dict=feed_dict_)
                y1s.append(y1)
                y2s.append(y2)
                sum_loss_val += loss_value
                i_batch += 1
            except tf.errors.OutOfRangeError:
                y1s = np.concatenate(y1s)
                y2s = np.concatenate(y2s)
                answer_dict, _, noanswer_num = convert_tokens(eval_file, dev_qid.tolist(), y1s.tolist(),
                                                              y2s.tolist(), data_type=2)
                metrics = evaluate(eval_file, answer_dict)
                ems.append(metrics['exact_match'])
                f1s.append(metrics['f1'])

                if metrics['f1'] < f1s[-1]:
                    config['learning_rate'] *= 0.5
                    logger.warning('learning rate reduce to:' + str(config['learning_rate']))
                    if config['learning_rate'] <= 1e-4:
                        logger.warning('rl loss start...')
                        config['rlw'] = config['rlw2']

                if ems[-1] > best_em:
                    best_em = ems[-1]
                if f1s[-1] > best_f1:
                    best_f1 = f1s[-1]
                logger.warning("-loss: %.4f -EM:%.2f%% (best: %.2f%%), -F1:%.2f%% (best: %.2f%%) -Noanswer:%d" %
                               (sum_loss_val / (i_batch + 1), metrics['exact_match'], best_em, metrics['f1'],
                                best_f1, noanswer_num))
                metrics = evaluate_acc(eval_file, answer_dict)
                logger.warning("Has answer acc:%.2f%%, No answer acc::%.2f%%" % (
                    metrics['has_answer_acc'] * 100, metrics['hasno_answer_acc'] * 100))

                break
