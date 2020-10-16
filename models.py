import os

import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel

from timit import TimitData

TIMIT_DIR='timit/TIMIT'

class Wav2VecData:

    def get_in_c_time(time):
        ratio = 0.006196484134864083
        return int(ratio*time)

    def calculate_C_wav2vec(self, filename):
        wav_input_16khz, sr = torchaudio.load(filename)
        if sr != 16000:
            raise Exception("Sample rate is {}, please resample to be 16 kHz".format(sr / 1000))
        z = self.model.feature_extractor(wav_input_16khz)
        c = self.model.feature_aggregator(z).detach().numpy()
        return c

    def __init__(self, wav2vec_model='/home/michael/Documents/Cogmaster/M1/S1/stage/wav2vec_large.pt',
            cache_file='timitdata.ft'):

        cp = torch.load(wav2vec_model, map_location=torch.device('cpu'))
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()

        if not os.path.exists(cache_file):
            wav2vec_timit = TimitData(self.calculate_C_wav2vec,
                    Wav2VecData.get_in_c_time,
                    timit_dir='{}/train/'.format(TIMIT_DIR),
                    dropna=True,
                    save_file=cache_file)
        else:
            wav2vec_timit = TimitData(load_file=cache_file)
        test_file = cache_file.replace('.ft', '_test.ft')
        if not os.path.exists(test_file):
            wav2vec_timit_test = TimitData(self.calculate_C_wav2vec,
                    Wav2VecData.get_in_c_time,
                    timit_dir='{}/test/'.format(TIMIT_DIR),
                    dropna=True,
                    save_file=test_file)
        else:
            wav2vec_timit_test = TimitData(load_file=test_file)

        self.test = wav2vec_timit_test
        self.train = wav2vec_timit
