from .model import ModelData
from .CPCData import CPCData
from .VQWav2VecData import VQWav2VecData
from .Wav2VecData import Wav2VecData

import logging
logging.getLogger("fairseq").setLevel(logging.WARNING)
