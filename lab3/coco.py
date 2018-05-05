import h5py
import json
import os
import numpy as np

import torch
import torch.utils.data as data

class CocoCaptionsFeature(data.Dataset):
    @property
    def dictionary(self):
        return self.info['ix_to_word']

    @property
    def seq_len(self):
        return self.label['labels'].shape[1]

    def __init__(self, fc_dir, att_dir, label_file, info_file, opt, split=None):

        self.seq_per_img = opt.seq_per_img

        self.feats_fc = h5py.File(os.path.join(fc_dir, 'feats_fc.h5'), 'r', libver='latest')
        self.feats_att = h5py.File(os.path.join(att_dir, 'feats_att.h5'), 'r', libver='latest')
        self.label = h5py.File(label_file, 'r', libver='latest')

        with open(info_file) as f:
            self.info = json.load(f)

        if isinstance(split, str):
            split = [split]

        assert split is None or isinstance(split, list)

        if not split is None:
            self.info['images'] = list(filter(lambda img: img['split'] in split, self.info['images']))


    def __getitem__(self, index):
        assert isinstance(index, int), '"index" must be "int"'

        ix = str(self.info['images'][index]['id'])
        fc = torch.from_numpy(np.array(self.feats_fc[ix]).astype('float32'))
        att = torch.from_numpy(np.array(self.feats_att[ix]).astype('float32'))

        # label_ix is 1-based, but label is 0-based
        labels_idx = slice(self.label['label_start_ix'][index]-1,
                          self.label['label_end_ix'][index])

        labels_origin = torch.from_numpy(self.label['labels'][labels_idx].astype('int32'))
        labels_n = labels_origin.shape[0]

        if labels_n <= self.seq_per_img:
            repeat_n = (self.seq_per_img - 1) // labels_n + 1
            labels = torch.cat([labels_origin] * repeat_n, dim=0)[:self.seq_per_img]
        else:
            choice = torch.randperm(labels_origin.shape[0])[:self.seq_per_img]
            labels = labels_origin[choice]

        return (fc, att, labels)
    
    def __len__(self):
        return len(self.info['images'])
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.feats_fc.close()
        self.feats_att.close()
        self.label.close()
