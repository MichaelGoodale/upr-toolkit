import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel

from .model import ModelData

class VQWav2VecData(ModelData):
    def get_in_c_time(self, time):
        ratio = 0.006196484134864083
        return int(ratio*time)

    def calculate_c(self, filename):
        wav_input_16khz, sr = torchaudio.load(filename)
        if sr != 16000:
            raise Exception("Sample rate is {}, please resample to be 16 kHz".format(sr / 1000))
        z = self.model.feature_extractor(wav_input_16khz)
        _, idxs = self.model.vector_quantizer.forward_idx(z)
        idxs = idxs.detach().numpy().swapaxes(1,2)
        return idxs

    def __init__(self, wav2vec_model='/home/michael/Documents/Cogmaster/M1/S1/stage/vq-wav2vec.pt',
            cache_file='/home/michael/Documents/Cogmaster/M1/S1/stage/model_caches/vq_wav2vec.ft',
            max_files=None):

        cp = torch.load(wav2vec_model, map_location=torch.device('cpu'))
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()
        super().__init__(cache_file, max_files=max_files)
