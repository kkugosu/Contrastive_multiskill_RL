import numpy as np
import random


class CustomDataLoader:

    def __init__(self, dataset, batch_size, suffle = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.suffle = suffle
        self.iternum = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.num_category = len(self.dataset[0])
        if self.suffle == True:
            index = random.sample(list(range(len(self.dataset))), self.batch_size)
        else:
            index = [i + self.iternum * self.batch_size for i in range(self.batch_size)]
        _batchdata = np.array(self.dataset[index[0]], dtype=np.object)
        _cnum = 0
        while _cnum < self.num_category:
            i = 0
            i = i + 1
            _batchdata[_cnum] = np.array([_batchdata[_cnum]])
            while i < self.batch_size:
                added_data = np.array([self.dataset[index[i]][_cnum]])
                _batchdata[_cnum] = np.concatenate((_batchdata[_cnum], added_data), axis=0)
                i = i + 1
            _cnum = _cnum + 1
        self.iternum = self.iternum + 1
        if (self.iternum + 1) * self.batch_size > len(self.dataset):
            self.iternum = 0
        return _batchdata

