import torch
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

