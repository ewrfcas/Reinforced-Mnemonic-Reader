import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
import h5py

data_type = 'train'
data_source = 'dataset_pre3'

# load trainset
qid = np.load(data_source + '/' + data_type + '_qid.npy').astype(np.int32)
print(data_type + 'data loading over...')

length = qid.shape[0]
print(length)
index = [i for i in range(0, length)]
random.shuffle(index)
print(index[0:10])

qid = qid[index]
tfrecords_filename = 'tfrecords/' + data_type + '_pre_elmo_cove3.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

with h5py.File(data_source + '/train_ELMO_feats.h5', 'r') as f1:
    with h5py.File(data_source + '/train_COVE_feats.h5', 'r') as f2:
        with h5py.File(data_source + '/train_data.h5', 'r') as f:
            for i in tqdm(range(len(qid))):
                elmo_context_feat = f1[str(qid[i]) + 'c'][:]
                elmo_question_feat = f1[str(qid[i]) + 'q'][:]
                cove_context_feat = f2[str(qid[i]) + 'c'][:]
                cove_question_feat = f2[str(qid[i]) + 'q'][:]

                data_simple = f[str(qid[i])]
                context_ids = data_simple['context_ids'][:]
                ques_ids = data_simple['ques_ids'][:]
                context_char_ids = data_simple['context_char_ids'][:]
                ques_char_ids = data_simple['ques_char_ids'][:]
                y1 = data_simple['y1'][:]
                y2 = data_simple['y2'][:]
                y1p = data_simple['y1p'][:]
                y2p = data_simple['y2p'][:]
                context_feat = data_simple['context_feat'][:]
                ques_feat = data_simple['ques_feat'][:]

                record = tf.train.Example(features=tf.train.Features(feature={
                    "context_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_ids.tostring()])),
                    "ques_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_ids.tostring()])),
                    "context_char_ids": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[context_char_ids.tostring()])),
                    "ques_char_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_ids.tostring()])),
                    "context_feat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_feat.tostring()])),
                    "ques_feat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_feat.tostring()])),
                    'elmo_context_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[elmo_context_feat.tostring()])),
                    'elmo_question_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[elmo_question_feat.tostring()])),
                    'cove_context_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[cove_context_feat.tostring()])),
                    'cove_question_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[cove_question_feat.tostring()])),
                    "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                    "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                    "y1p": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1p.tostring()])),
                    "y2p": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2p.tostring()])),
                    "qid": tf.train.Feature(int64_list=tf.train.Int64List(value=[qid[i]]))
                }))
                writer.write(record.SerializeToString())
writer.close()

data_type = 'dev'
data_source = 'dataset_pre3'

# load trainset
qid = np.load(data_source + '/' + data_type + '_qid.npy').astype(np.int32)
print(data_type + 'data loading over...')

tfrecords_filename = 'tfrecords/' + data_type + '_pre_elmo_cove3.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

with h5py.File(data_source + '/dev_ELMO_feats.h5', 'r') as f1:
    with h5py.File(data_source + '/dev_COVE_feats.h5', 'r') as f2:
        with h5py.File(data_source + '/dev_data.h5', 'r') as f:
            for i in tqdm(range(len(qid))):
                elmo_context_feat = f1[str(qid[i]) + 'c'][:]
                elmo_question_feat = f1[str(qid[i]) + 'q'][:]
                cove_context_feat = f2[str(qid[i]) + 'c'][:]
                cove_question_feat = f2[str(qid[i]) + 'q'][:]

                data_simple = f[str(qid[i])]
                context_ids = data_simple['context_ids'][:]
                ques_ids = data_simple['ques_ids'][:]
                context_char_ids = data_simple['context_char_ids'][:]
                ques_char_ids = data_simple['ques_char_ids'][:]
                y1 = data_simple['y1'][:]
                y2 = data_simple['y2'][:]
                y1p = data_simple['y1p'][:]
                y2p = data_simple['y2p'][:]
                context_feat = data_simple['context_feat'][:]
                ques_feat = data_simple['ques_feat'][:]

                record = tf.train.Example(features=tf.train.Features(feature={
                    "context_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_ids.tostring()])),
                    "ques_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_ids.tostring()])),
                    "context_char_ids": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[context_char_ids.tostring()])),
                    "ques_char_ids": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_char_ids.tostring()])),
                    "context_feat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_feat.tostring()])),
                    "ques_feat": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_feat.tostring()])),
                    'elmo_context_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[elmo_context_feat.tostring()])),
                    'elmo_question_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[elmo_question_feat.tostring()])),
                    'cove_context_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[cove_context_feat.tostring()])),
                    'cove_question_feat': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[cove_question_feat.tostring()])),
                    "y1": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1.tostring()])),
                    "y2": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2.tostring()])),
                    "y1p": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y1p.tostring()])),
                    "y2p": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y2p.tostring()])),
                    "qid": tf.train.Feature(int64_list=tf.train.Int64List(value=[qid[i]]))
                }))
                writer.write(record.SerializeToString())
writer.close()
