import os

import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel

from utils import PretrainedCPCModel
from timit import TimitData

TIMIT_DIR='timit/TIMIT'

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


class Wav2VecData(ModelData):

    def get_in_c_time(self, time):
        ratio = 0.006196484134864083
        return int(ratio*time)

    def calculate_c(self, filename):
        wav_input_16khz, sr = torchaudio.load(filename)
        if sr != 16000:
            raise Exception("Sample rate is {}, please resample to be 16 kHz".format(sr / 1000))
        z = self.model.feature_extractor(wav_input_16khz)
        c = self.model.feature_aggregator(z).detach().numpy()
        return c

    def __init__(self, wav2vec_model='/home/michael/Documents/Cogmaster/M1/S1/stage/wav2vec_large.pt',
            cache_file='timitdata.ft',
            max_files=None):

        cp = torch.load(wav2vec_model, map_location=torch.device('cpu'))
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()
        super().__init__(cache_file, max_files=max_files)

class CPCData(ModelData):

    def get_in_c_time(self, time):
        ratio = 0.006196484134864083
        return int(ratio*time)

    def calculate_c(self, filename):
        wav_input_16khz, sr = torchaudio.load(filename)
        if sr != 16000:
            raise Exception("Sample rate is {}, please resample to be 16 kHz".format(sr / 1000))
        encodedData, cFeature, _ = model(wav_input_16khz)
        return encodedData 

    def __init__(self, cpc_model='/home/michael/Documents/Cogmaster/M1/S1/stage/CPC/michael_pretrained/english_model/checkpoint_60.pt',
            cache_file='cpc_eng_data.ft',
            max_files=None):

        cp = torch.load(wav2vec_model, map_location=torch.device('cpu'))
        self.model = PretrainedCPCModel(cpc_model)
        self.model.eval()
        super.__init__(cache_file, max_files=max_files)
