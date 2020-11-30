import torch
import torchaudio
import numpy as np
from fairseq.models.wav2vec import Wav2VecModel, Wav2Vec2Model

from .model import ModelData

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
            cache_file='/home/michael/Documents/Cogmaster/M1/S1/stage/model_caches/timitdata.ft',
            max_files=None):

        cp = torch.load(wav2vec_model, map_location=torch.device('cpu'))
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()
        super().__init__(cache_file, max_files=max_files)

class Wav2Vec2Data(ModelData):
    def get_in_c_time(self, time):
        ratio = 0.001552557134626
        return int(ratio*time)

    def calculate_c(self, filename):
        wav_input_16khz, sr = torchaudio.load(filename)
        if sr != 16000:
            raise Exception("Sample rate is {}, please resample to be 16 kHz".format(sr / 1000))
        #c = self.model(wav_input_16khz)
        #c = c['x'].detach().numpy()
        #c = np.swapaxes(c, 0, 1)
        c = self.model.feature_extractor(wav_input_16khz)
        print(c)
        c[np.isinf(c)] = 0.0
        c[np.isnan(c)] = 0.0
        return c

    def __init__(self, wav2vec_model='/home/michael/Documents/Cogmaster/M1/S1/stage/libri960_big.pt',
            cache_file='/home/michael/Documents/Cogmaster/M1/S1/stage/model_caches/wav2vecBIG.ft',
            max_files=None):

        cp = torch.load(wav2vec_model, map_location=torch.device('cpu'))
        self.model = Wav2Vec2Model.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()
        super().__init__(cache_file, max_files=max_files)
