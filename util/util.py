import re
from collections import Counter
import string
import time
import tensorflow as tf

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def get_record_parser(config):
    def parser(example):
        if not config['data_type']:
            config['data_type'] = 2
        char_limit = config['char_limit']
        features_ = {
            "context_ids": tf.FixedLenFeature([], tf.string),
            "ques_ids": tf.FixedLenFeature([], tf.string),
            "context_char_ids": tf.FixedLenFeature([], tf.string),
            "ques_char_ids": tf.FixedLenFeature([], tf.string),
            'context_feat': tf.FixedLenFeature([], tf.string),
            'ques_feat': tf.FixedLenFeature([], tf.string),
            'elmo_context_feat': tf.FixedLenFeature([], tf.string),
            'elmo_question_feat': tf.FixedLenFeature([], tf.string),
            'cove_context_feat': tf.FixedLenFeature([], tf.string),
            'cove_question_feat': tf.FixedLenFeature([], tf.string),
            "y1": tf.FixedLenFeature([], tf.string),
            "y2": tf.FixedLenFeature([], tf.string),
            "qid": tf.FixedLenFeature([], tf.int64)
        }
        if config['data_type'] == 2:
            features_['y1p'] = tf.FixedLenFeature([], tf.string)
            features_['y2p'] = tf.FixedLenFeature([], tf.string)

        features = tf.parse_single_example(example, features=features_)
        context_idxs = tf.reshape(tf.decode_raw(features["context_ids"], tf.int32), [-1])
        ques_idxs = tf.reshape(tf.decode_raw(features["ques_ids"], tf.int32), [-1])
        context_char_idxs = tf.reshape(tf.decode_raw(features["context_char_ids"], tf.int32), [-1, char_limit])
        ques_char_idxs = tf.reshape(tf.decode_raw(features["ques_char_ids"], tf.int32), [-1, char_limit])
        context_feat = tf.reshape(tf.decode_raw(features["context_feat"], tf.float32), [-1, 73])
        ques_feat = tf.reshape(tf.decode_raw(features["ques_feat"], tf.float32), [-1, 73])
        elmo_context_feat = tf.reshape(tf.decode_raw(features['elmo_context_feat'], tf.float32), [-1, 3, 1024])
        elmo_question_feat = tf.reshape(tf.decode_raw(features['elmo_question_feat'], tf.float32), [-1, 3, 1024])
        cove_context_feat = tf.reshape(tf.decode_raw(features['cove_context_feat'], tf.float32), [-1, 2, 600])
        cove_question_feat = tf.reshape(tf.decode_raw(features['cove_question_feat'], tf.float32), [-1, 2, 600])
        y1 = tf.reshape(tf.decode_raw(features["y1"], tf.int32), [-1])
        y2 = tf.reshape(tf.decode_raw(features["y2"], tf.int32), [-1])
        if config['data_type'] == 2:
            y1p = tf.reshape(tf.decode_raw(features["y1p"], tf.int32), [-1])
            y2p = tf.reshape(tf.decode_raw(features["y2p"], tf.int32), [-1])
        # qid = features["qid"]
        if config['data_type'] == 2:
            return context_idxs, ques_idxs, \
                   context_char_idxs, ques_char_idxs, \
                   context_feat, ques_feat, \
                   elmo_context_feat, elmo_question_feat, \
                   cove_context_feat, cove_question_feat, \
                   y1, y2, y1p, y2p
        else:
            return context_idxs, ques_idxs, \
                   context_char_idxs, ques_char_idxs, \
                   context_feat, ques_feat, \
                   elmo_context_feat, elmo_question_feat, \
                   cove_context_feat, cove_question_feat, \
                   y1, y2

    return parser


def convert_tokens(eval_file, qa_id, pp1, pp2, unanswer_id=-1, data_type=2):
    answer_dict = {}
    remapped_dict = {}
    noanswer_num = 0
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        if data_type == 2:
            if p1 == unanswer_id or p2 == unanswer_id or p1 >= len(spans) or p2 >= len(
                    spans):  # prediction has no answer
                noanswer_num += 1
                answer_dict[str(qid)] = ''
                remapped_dict[uuid] = ''
            else:
                start_idx = spans[min(p1, len(spans) - 1)][0]
                end_idx = spans[min(p2, len(spans) - 1)][1]
                answer_dict[str(qid)] = context[start_idx: end_idx]
                remapped_dict[uuid] = context[start_idx: end_idx]
        else:
            start_idx = spans[min(p1, len(spans) - 1)][0]
            end_idx = spans[min(p2, len(spans) - 1)][1]
            answer_dict[str(qid)] = context[start_idx: end_idx]
            remapped_dict[uuid] = context[start_idx: end_idx]
    return answer_dict, remapped_dict, noanswer_num


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        prediction = value
        if len(ground_truths) == 0:  # ground truth has no answer
            if prediction == '':
                exact_match += 1
                f1 += 1
        else:
            exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
            f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def evaluate_acc(eval_file, answer_dict):
    has_answer_acc = 0
    has_answer_total = 0
    hasno_answer_acc = 0
    hasno_answer_total = 0
    for key, value in answer_dict.items():
        ground_truths = eval_file[key]["answers"]
        prediction = value
        if len(ground_truths) != 0:  # ground truth has answers
            has_answer_total += 1
            if prediction != '':
                has_answer_acc += 1
        else:
            hasno_answer_total += 1
            if prediction == '':
                hasno_answer_acc += 1
    print(has_answer_acc, '/', has_answer_total, hasno_answer_acc, '/', hasno_answer_total)
    has_answer_acc /= has_answer_total
    hasno_answer_acc /= hasno_answer_total
    return {'has_answer_acc': has_answer_acc, 'hasno_answer_acc': hasno_answer_acc}


def evaluate_max(eval_file, answer_dict_list):
    f1 = exact_match = total = 0
    for key, value in answer_dict_list[0].items():
        total += 1
        ground_truths = eval_file[key]["answers"]
        f1_temp = 0
        em_temp = 0
        for answer_dict in answer_dict_list:
            prediction = answer_dict[key]
            if len(ground_truths) == 0:  # ground truth has no answer
                if prediction == 'unanswerable':
                    em_temp = 1
                    f1_temp = 1
            else:
                em_temp = max(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths), em_temp)
                f1_temp = max(metric_max_over_ground_truths(f1_score, prediction, ground_truths), f1_temp)
        exact_match += em_temp
        f1 += f1_temp
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return {'exact_match': exact_match, 'f1': f1}


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def cal_ETA(t_start, i, n_batch):
    t_temp = time.time()
    t_avg = float(int(t_temp) - int(t_start)) / float(i + 1)
    if n_batch - i - 1 > 0:
        return int((n_batch - i - 1) * t_avg)
    else:
        return int(t_temp) - int(t_start)


import numpy as np
import h5py


def batchify_train(data):
    def padding(datas):
        max_len = max([d.shape[0] for d in datas])
        paded_datas = np.zeros([len(datas), max_len] + list(datas[0].shape[1:]), dtype=datas[0].dtype)
        for i in range(len(datas)):
            paded_datas[i, 0:datas[i].shape[0]] = datas[i]
        return paded_datas

    cont_ids, cont_char_ids, ques_ids, ques_char_ids = [], [], [], []
    cont_feat, ques_feat, y1s, y2s, y1ps, y2ps = [], [], [], [], [], []
    elmo_cont_feat, elmo_ques_feat = [], []
    cove_cont_feat, cove_ques_feat = [], []
    # load elmo
    with h5py.File('dataset_pre3/train_ELMO_feats.h5', 'r') as elmo_h5f:
        with h5py.File('dataset_pre3/train_COVE_feats.h5', 'r') as cove_h5f:
            with h5py.File('dataset_pre3/train_data.h5', 'r') as h5f:
                for qid in data:
                    group = h5f[str(qid)]
                    # base feats
                    cont_ids.append(group['context_ids'][:])
                    cont_char_ids.append(group['context_char_ids'][:])
                    cont_feat.append(group['context_feat'][:])
                    ques_ids.append(group['ques_ids'][:])
                    ques_char_ids.append(group['ques_char_ids'][:])
                    ques_feat.append(group['ques_feat'][:])
                    # elmo feats
                    elmo_cont_feat.append(elmo_h5f[str(qid) + 'c'][:])
                    elmo_ques_feat.append(elmo_h5f[str(qid) + 'q'][:])
                    # cove feats
                    cove_cont_feat.append(cove_h5f[str(qid) + 'c'][:])
                    cove_ques_feat.append(cove_h5f[str(qid) + 'q'][:])
    cont_ids = padding(cont_ids)
    cont_char_ids = padding(cont_char_ids)
    ques_ids = padding(ques_ids)
    ques_char_ids = padding(ques_char_ids)
    elmo_cont_feat = padding(elmo_cont_feat)
    elmo_ques_feat = padding(elmo_ques_feat)
    cove_cont_feat = padding(cove_cont_feat)
    cove_ques_feat = padding(cove_ques_feat)

    return cont_ids, cont_char_ids, ques_ids, ques_char_ids, elmo_cont_feat, elmo_ques_feat, cove_cont_feat, cove_ques_feat


def batchify_dev(data):
    def padding(datas):
        max_len = max([d.shape[0] for d in datas])
        paded_datas = np.zeros([len(datas), max_len] + list(datas[0].shape[1:]), dtype=datas[0].dtype)
        for i in range(len(datas)):
            paded_datas[i, 0:datas[i].shape[0]] = datas[i]
        return paded_datas

    cont_ids, cont_char_ids, ques_ids, ques_char_ids = [], [], [], []
    cont_feat, ques_feat, y1s, y2s, y1ps, y2ps = [], [], [], [], [], []
    elmo_cont_feat, elmo_ques_feat = [], []
    cove_cont_feat, cove_ques_feat = [], []
    # load elmo
    with h5py.File('dataset_pre3/dev_ELMO_feats.h5', 'r') as elmo_h5f:
        with h5py.File('dataset_pre3/dev_COVE_feats.h5', 'r') as cove_h5f:
            with h5py.File('dataset_pre3/dev_data.h5', 'r') as h5f:
                for qid in data:
                    group = h5f[str(qid)]
                    # base feats
                    cont_ids.append(group['context_ids'][:])
                    cont_char_ids.append(group['context_char_ids'][:])
                    cont_feat.append(group['context_feat'][:])
                    ques_ids.append(group['ques_ids'][:])
                    ques_char_ids.append(group['ques_char_ids'][:])
                    ques_feat.append(group['ques_feat'][:])
                    # elmo feats
                    elmo_cont_feat.append(elmo_h5f[str(qid) + 'c'][:])
                    elmo_ques_feat.append(elmo_h5f[str(qid) + 'q'][:])
                    # cove feats
                    cove_cont_feat.append(cove_h5f[str(qid) + 'c'][:])
                    cove_ques_feat.append(cove_h5f[str(qid) + 'q'][:])
    cont_ids = padding(cont_ids)
    cont_char_ids = padding(cont_char_ids)
    ques_ids = padding(ques_ids)
    ques_char_ids = padding(ques_char_ids)
    elmo_cont_feat = padding(elmo_cont_feat)
    elmo_ques_feat = padding(elmo_ques_feat)
    cove_cont_feat = padding(cove_cont_feat)
    cove_ques_feat = padding(cove_ques_feat)

    return cont_ids, cont_char_ids, ques_ids, ques_char_ids, elmo_cont_feat, elmo_ques_feat, cove_cont_feat, cove_ques_feat
