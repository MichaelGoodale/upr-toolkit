import os

import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel

from upr_toolkit.timit import TimitData


TIMIT_DIR='/home/michael/Documents/Cogmaster/M1/S1/stage/timit/TIMIT'

class ModelData:
    def get_in_c_time(self, time):
        raise NotImplementedError("get_in_c_time must be defined in a subclass")

    def calculate_c(self, filename):
        raise NotImplementedError("calculate_c must be defined in a subclass")

    def __init__(self, cache_file, max_files=None):
        if not os.path.exists(cache_file):
            timit = TimitData(self.calculate_c,
                    self.get_in_c_time,
                    timit_dir='{}/train/'.format(TIMIT_DIR),
                    max_files=max_files,
                    dropna=True,
                    save_file=cache_file)
        else:
            timit = TimitData(load_file=cache_file)

        test_file = cache_file.replace('.ft', '_test.ft')
        if not os.path.exists(test_file):
            timit_test = TimitData(self.calculate_c,
                    self.get_in_c_time,
                    timit_dir='{}/test/'.format(TIMIT_DIR),
                    max_files=max_files,
                    dropna=True,
                    save_file=test_file)
        else:
            timit_test = TimitData(load_file=test_file)

        self.train = timit
        self.test = timit_test
