import os

import torch
import torchaudio

from .model import ModelData
from cpc.feature_loader import loadModel

class PretrainedCPCModel(torch.nn.Module):

    """ Class that handles CPC pretrained models
    https://github.com/facebookresearch/CPC_audio
    """
    def __init__(self, model_path, intermediate_idx=0):
        super().__init__()
        self.model = loadModel([model_path], intermediate_idx=intermediate_idx)[0]
        self.model.gAR.keepHidden = True

    def forward(self, x):
        with torch.no_grad():
            x = x.view(1, 1, -1)
            encodedData = self.model.gEncoder(x).permute(0, 2, 1)
            cFeature = self.model.gAR(encodedData)
            encodedData = encodedData.permute(0, 2, 1)
            cFeature = cFeature.permute(0, 2, 1)
            return encodedData, cFeature, None


class CPCData(ModelData):

    def get_in_c_time(self, time):
        ratio = 0.00623888282
        return int(ratio*time)

    def calculate_c(self, filename):
        wav_input_16khz, sr = torchaudio.load(filename)
        if sr != 16000:
            raise Exception("Sample rate is {}, please resample to be 16 kHz".format(sr / 1000))
        _, cFeature, _ = self.model(wav_input_16khz)
        return cFeature.detach().numpy()

    def __init__(self, cpc_model='/home/michael/Documents/Cogmaster/M1/S1/stage/CPC/michael_pretrained/english_model/checkpoint_60.pt',
            cache_file='/home/michael/Documents/Cogmaster/M1/S1/stage/model_caches/cpc_eng_data.ft',
            max_files=None):

        self.model = PretrainedCPCModel(model_path=cpc_model)
        self.model.eval()
        super().__init__(cache_file, max_files=max_files)
