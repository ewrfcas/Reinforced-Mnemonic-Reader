import os
import json
import RMR_modelV6 as RMR
import tensorflow.contrib.slim as slim
from util.util import *
import tensorflow as tf
import pandas as pd
from util.h5py_generator import Generator
from util.log_wrapper import create_logger

os.environ["CUDA_VISIBLE_DEVICES"] = '4'

if __name__ == '__main__':

    data_source = 'dataset'

    config = {
        'char_dim': 300,
        'cont_limit': 400,
        'ques_limit': 50,
        'char_limit': 16,
        'ans_limit': -1,
        'filters': 128,
        'char_filters': 100,
        'dropout': 0.175,
        'dropout_emb': 0.15,
        'dropout_att': 0.2,
        'dropout_rnn': 0.15,
        'l2_norm': 3e-7,
        'decay': 1,
        'gamma_b': 0.3,
        'gamma_c': 1.0,
        'init_lambda': 3.0,
        'learning_rate': 1e-3,
        'shuffle_size': 25000,
        'grad_clip': 5.0,
        'use_elmo': 0,
        'use_cove': 0,
        'use_feat': True,
        'use_rlloss': False,
        'rlw': 0.0,
        'rlw2': 0.8,
        'optimizer': 'adam',
        'cove_path': '../SAN_tf/Keras_CoVe_2layers.h5',
        'elmo_weights_path': '../SAN_tf/elmo_tf/models/squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5',
        'elmo_options_path': '../SAN_tf/elmo_tf/models/options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json',
        'train_tfrecords': '../QANet_tf/tfrecords/train_pre_elmo_cove3.tfrecords',
        'dev_tfrecords': '../QANet_tf/tfrecords/dev_pre_elmo_cove3.tfrecords',
        'batch_size': 32,
        'epoch': 30,
        'origin_path': None,  # not finetune
        'path': 'RMRV102'
    }

    global logger
    logger = create_logger(__name__, to_disk=True, log_file='log/' + config['path'] + '.log')

    logger.info('loading data...')
    train_qid = np.load(data_source + '/train_qid.npy').astype(np.int32)
    dev_qid = np.load(data_source + '/dev_qid.npy').astype(np.int32)
    with open(data_source + '/test_eval.json', "r") as fh:
        eval_file = json.load(fh)

    # load embedding matrix
    logger.info('loading embedding...')
    word_mat = np.load(data_source + '/word_emb_mat.npy')
    char_mat_fix = np.load(data_source + '/char_emb_mat_fix.npy').astype(np.float32)
    char_mat_trainable = np.load(data_source + '/char_emb_mat_trainable.npy').astype(np.float32)

    logger.info('init model...')
    model = RMR.Model(config, word_mat=word_mat, char_mat_trainable=char_mat_trainable, char_mat_fix=char_mat_fix)
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    best_f1 = 0
    best_em = 0
    f1s = []
    ems = []

    logger.info('init generator...')
    train_gen = Generator(data_source + '/train_data.h5', train_qid, batch_size=config['batch_size'], shuffle=True,
                          use_elmo=config['use_elmo'], use_cove=config['use_cove'],
                          elmo_path=data_source + '/train_ELMO_feats.h5',
                          cove_path=data_source + '/train_COVE_feats.h5')
    dev_gen = Generator(data_source + '/dev_data.h5', dev_qid, batch_size=config['batch_size'], shuffle=False,
                        use_elmo=config['use_elmo'], use_cove=config['use_cove'],
                        elmo_path=data_source + '/dev_ELMO_feats.h5', cove_path=data_source + '/dev_COVE_feats.h5')

    logger.info('starting session...')
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

        use_rl=False
        for i_epoch in range(config['epoch']):
            if (i_epoch + 1) % 8 == 0:
                config['learning_rate'] *= 0.5
            #     logger.warning('learning rate reduce to:' + str(config['learning_rate']))
            #     if config['learning_rate'] <= 2.5e-4:
            #         use_rl=True
            #         logger.warning('rl loss start...')
            #         config['rlw'] = config['rlw2']

            sum_loss = 0
            for i_batch in range(train_gen.max_batch):
                assert i_batch == train_gen.i_batch
                # if use_rl:
                #     config['rlw'] = min(config['rlw2'], config['rlw']+config['rlw2']/5000)
                if i_batch == 1:
                    t_start = time.time()
                data_batch = next(train_gen)
                feed_dict_ = {model.contw_input: data_batch['context_ids'], model.quesw_input: data_batch['ques_ids'],
                              model.contc_input: data_batch['context_char_ids'],
                              model.quesc_input: data_batch['ques_char_ids'],
                              model.y_start: data_batch['y1'], model.y_end: data_batch['y2'],
                              # model.yp_start: data_batch['y1p'], model.yp_end: data_batch['y2p'],
                              model.un_size: data_batch['context_ids'].shape[0],
                              model.dropout: config['dropout'],
                              model.dropout_emb: config['dropout_emb'],
                              model.dropout_att: config['dropout_att'],
                              model.dropout_rnn: config['dropout_rnn'],
                              model.learning_rate: config['learning_rate'],
                              model.rlw: config['rlw']}
                if config['use_feat']:
                    feed_dict_[model.cont_feat] = data_batch['context_feat']
                    feed_dict_[model.ques_feat] = data_batch['ques_feat']
                if config['use_elmo'] == 1:
                    feed_dict_[model.elmo_cont] = data_batch['elmo_cont']
                    feed_dict_[model.elmo_ques] = data_batch['elmo_ques']
                if config['use_cove'] == 1:
                    feed_dict_[model.cove_cont] = data_batch['cove_cont']
                    feed_dict_[model.cove_ques] = data_batch['cove_ques']
                if config['decay'] < 1:
                    loss_value, _ = sess.run([model.loss, model.ema_train_op], feed_dict=feed_dict_)
                else:
                    loss_value, _ = sess.run([model.loss, model.train_op], feed_dict=feed_dict_)
                char_mat = sess.run(model.char_mat)
#                 ipdb.set_trace()
                char_mat[-char_mat_fix.shape[0]:,::] = char_mat_fix
                _ = sess.run(model.assign_char_mat, feed_dict={model.old_char_mat:char_mat})
                sum_loss += loss_value

                # # check embedding
                # fix_feat, tra_feat = sess.run([model.char_mat[-93:, :], model.char_mat[0:1140, :]])
                # fix_feat = np.sum(fix_feat)
                # tra_feat = np.sum(tra_feat)
                # print('fix:', fix_feat)
                # print('trainable:', tra_feat)

                last_train_str = "[epoch:%d/%d, steps:%d/%d] -loss:%.4f" % (i_epoch + 1, config['epoch'], i_batch + 1,
                                                                            train_gen.max_batch,
                                                                            sum_loss / (i_batch + 1))
                if i_batch > 0:
                    last_train_str += (' -ETA:%ds' % cal_ETA(t_start, i_batch, train_gen.max_batch))
                if i_batch % 100 == 0:
                    logger.info(last_train_str)
            logger.info(last_train_str)

            # validating step
            # # save the temp weights and do ema
            # if config['decay'] < 1.0:
            #     saver.save(sess, os.path.join('model', config['path'], 'temp_model.ckpt'))
            #     sess.run(model.assign_vars)
            #     print('EMA over...')
            logger.info('validating...')
            sum_loss_val = 0
            y1s = []
            y2s = []
            dev_gen.reset()
            for i_batch in range(dev_gen.max_batch):
                assert i_batch == dev_gen.i_batch
                data_batch = next(dev_gen)
                feed_dict_ = {model.contw_input: data_batch['context_ids'], model.quesw_input: data_batch['ques_ids'],
                              model.contc_input: data_batch['context_char_ids'],
                              model.quesc_input: data_batch['ques_char_ids'],
                              model.y_start: data_batch['y1'], model.y_end: data_batch['y2'],
                              # model.yp_start: data_batch['y1p'], model.yp_end: data_batch['y2p'],
                              model.un_size: data_batch['context_ids'].shape[0]}
                if config['use_feat']:
                    feed_dict_[model.cont_feat] = data_batch['context_feat']
                    feed_dict_[model.ques_feat] = data_batch['ques_feat']
                if config['use_elmo'] == 1:
                    feed_dict_[model.elmo_cont] = data_batch['elmo_cont']
                    feed_dict_[model.elmo_ques] = data_batch['elmo_ques']
                if config['use_cove'] == 1:
                    feed_dict_[model.cove_cont] = data_batch['cove_cont']
                    feed_dict_[model.cove_ques] = data_batch['cove_ques']

                loss_value, y1, y2 = sess.run([model.loss, model.mask_output1, model.mask_output2],
                                              feed_dict=feed_dict_)
                y1s.append(y1)
                y2s.append(y2)
                sum_loss_val += loss_value

            y1s = np.concatenate(y1s)
            y2s = np.concatenate(y2s)
            answer_dict, _, noanswer_num = convert_tokens(eval_file, dev_qid.tolist(), y1s.tolist(),
                                                          y2s.tolist(), data_type=1)
            metrics = evaluate(eval_file, answer_dict)
            ems.append(metrics['exact_match'])
            f1s.append(metrics['f1'])

            # if metrics['f1'] < f1s[-1]:
            #     config['learning_rate'] *= 0.5
            #     logger.warning('learning rate reduce to:' + str(config['learning_rate']))
            #     if config['learning_rate'] <= 1e-4:
            #         logger.warning('rl loss start...')
            #         config['rlw'] = config['rlw2']

            if ems[-1] > best_em:
                best_em = ems[-1]
            if f1s[-1] > best_f1:
                best_f1 = f1s[-1]
            saver.save(sess, os.path.join('model', config['path'], 'model.ckpt'),
                       global_step=(i_epoch + 1) * train_gen.max_batch)
            logger.warning("-loss: %.4f -EM:%.2f%% (best: %.2f%%), -F1:%.2f%% (best: %.2f%%) -Noanswer:%d" %
                           (sum_loss_val / (dev_gen.max_batch + 1), metrics['exact_match'], best_em, metrics['f1'],
                            best_f1, noanswer_num))
#             metrics = evaluate_acc(eval_file, answer_dict)
#             logger.warning("Has answer acc:%.2f%%, No answer acc::%.2f%%" % (
#                 metrics['has_answer_acc'] * 100, metrics['hasno_answer_acc'] * 100))
            result = pd.DataFrame([ems, f1s], index=['em', 'f1']).transpose()
            result.to_csv('results/result_' + config['path'] + '.csv', index=None)

            # # recover the model
            # if config['decay'] < 1.0:
            #     saver.restore(sess, os.path.join('model', config['path'], 'temp_model.ckpt'))
            #     print('recover weights over...')
