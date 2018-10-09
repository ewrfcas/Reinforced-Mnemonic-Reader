import h5py
import numpy as np
import time

class Generator():
    def __init__(self, data_path, qid, batch_size=32, shuffle=True, padding_value=0, data_keys=None, use_elmo=0,
                 use_cove=0, elmo_path=None, cove_path=None):
        self.batch_size = batch_size
        if isinstance(qid, str):
            self.qid = np.load(qid)
        else:
            self.qid = qid
        self.length = len(self.qid)
        self.shuffle = shuffle
        self.data_path = data_path
        self.max_batch = self.length // self.batch_size
        if self.length % self.batch_size != 0:
            self.max_batch += 1
        self.padding_value = padding_value
        if self.shuffle:
            self.run_shuffle()
            print('Loaded {} samples'.format(self.length))
        self.i_batch = 0
        self.data_keys = data_keys
        self.use_elmo = use_elmo
        self.use_cove = use_cove
        if use_elmo == 1:
            assert elmo_path is not None
            self.elmo_path = elmo_path
        if use_cove == 1:
            assert cove_path is not None
            self.cove_path = cove_path

        self.get_time = 0
        self.pad_time = 0

    def reset(self):
        self.i_batch = 0
        self.run_shuffle()

    def run_shuffle(self):
        if self.shuffle:
            np.random.shuffle(self.qid)
        else:
            pass

    def padding(self, datas):
        max_len = max([d.shape[0] for d in datas])
        paded_datas = np.zeros([len(datas), max_len] + list(datas[0].shape[1:]), dtype=datas[0].dtype)
        for i in range(len(datas)):
            paded_datas[i, 0:datas[i].shape[0]] = datas[i]
        return paded_datas

    def __len__(self):
        return self.length

    def __next__(self):
        batch_data = {}
        if self.use_elmo == 1:
            elmo_h5f = h5py.File(self.elmo_path, 'r')
        if self.use_cove == 1:
            cove_h5f = h5py.File(self.cove_path, 'r')
        # st = time.time()
        with h5py.File(self.data_path, 'r') as h5f:
            qid_batch = self.qid[self.i_batch * self.batch_size:(self.i_batch + 1) * self.batch_size]
            for id in qid_batch:
                group = h5f[str(id)]
                # normal features
                if self.data_keys is None:
                    self.data_keys = list(group.keys())
                for k in self.data_keys:
                    if k not in batch_data:
                        batch_data[k] = [group[k][:]]
                    else:
                        batch_data[k].append(group[k][:])
                # elmo features
                if self.use_elmo == 1:
                    if 'elmo_cont' not in batch_data:
                        batch_data['elmo_cont'] = [elmo_h5f[str(id) + 'c'][:]]
                    else:
                        batch_data['elmo_cont'].append(elmo_h5f[str(id) + 'c'][:])
                    if 'elmo_ques' not in batch_data:
                        batch_data['elmo_ques'] = [elmo_h5f[str(id) + 'q'][:]]
                    else:
                        batch_data['elmo_ques'].append(elmo_h5f[str(id) + 'q'][:])
                # cove features
                if self.use_cove == 1:
                    if 'cove_cont' not in batch_data:
                        batch_data['cove_cont'] = [cove_h5f[str(id) + 'c'][:]]
                    else:
                        batch_data['cove_cont'].append(cove_h5f[str(id) + 'c'][:])
                    if 'cove_ques' not in batch_data:
                        batch_data['cove_ques'] = [cove_h5f[str(id) + 'q'][:]]
                    else:
                        batch_data['cove_ques'].append(cove_h5f[str(id) + 'q'][:])
        if self.use_elmo == 1:
            elmo_h5f.close()
        if self.use_cove == 1:
            cove_h5f.close()
        # ed = time.time()
        # self.get_time += float(ed - st)

        # st = time.time()
        for k in batch_data:
            batch_data[k] = self.padding(batch_data[k])
        # ed = time.time()
        # self.pad_time += float(ed - st)
        # print('get_time:', self.get_time)
        # print('pad_time:', self.pad_time)
        self.i_batch += 1
        if self.i_batch == self.max_batch:
            self.i_batch = 0
            self.run_shuffle()
        return batch_data
