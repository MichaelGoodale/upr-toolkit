import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torchaudio
from fairseq.models.wav2vec import Wav2VecModel

from timit import TimitData

def get_in_c_time(time):
    ratio = 0.006196484134864083
    return int(ratio*time)

def calculate_C_wav2vec(filename):
    wav_input_16khz, sr = torchaudio.load(filename)
    if sr != 16000:
        raise Exception("Sample rate is {}, please resample to be 16 kHz".format(sr / 1000))
    z = model.feature_extractor(wav_input_16khz)
    c = model.feature_aggregator(z)
    return c.detach().numpy()

cp = torch.load('/home/michael/Documents/Cogmaster/M1/S1/stage/wav2vec_large.pt', map_location=torch.device('cpu'))
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

wav2vec_timit = TimitData(calculate_C_wav2vec, get_in_c_time, dropna=True)

reducer = umap.UMAP(n_neighbors=25, min_dist=0.25, metric="cosine")
X = np.vstack(wav2vec_timit.phones_df["c"])
y = wav2vec_timit.phones_df["phone"]
print(X.shape)
from sklearn.svm import SVC
model = SVC(kernel='linear')
model.fit(X, y)
top_n = pd.Series(abs(model.coef_[0])).nlargest(50)


from sklearn.manifold import TSNE
embedding = TSNE(n_components=2).fit_transform(X[:, top_n.index])
color_map = sns.color_palette(n_colors=len(TimitData.PHONES))
sns.scatterplot(embedding[:, 0], embedding[:, 1], hue=wav2vec_timit.phones_df["phone"], alpha=0.5)
plt.gca().set_aspect('equal', 'datalim')
plt.legend()
plt.show()
