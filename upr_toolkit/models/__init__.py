from .model import ModelData

#TODO: This is horrible, find a better way to optionally import
try:
    from .CPCData import CPCData
except ImportError:
    pass
try:
    from .VQWav2VecData import VQWav2VecData
except ImportError:
    pass
try:
    from .Wav2VecData import Wav2VecData, Wav2Vec2Data
except ImportError:
    pass
try:
    from .RandomData import RandomData
except ImportError:
    pass

import logging
logging.getLogger("fairseq").setLevel(logging.WARNING)
