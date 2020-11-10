import os
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from timit import TimitData
from models import Wav2VecData, CPCData
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import linear_model

def get_formant_regression(train):
    idx = train.phones_df["phone"].isin(TimitData.VOWELS)
    phones = train.phones_df[idx]
    X = np.vstack(phones["c"])
    f1_y = phones["f1"]
    f2_y = phones["f2"]
    reg1 = linear_model.LinearRegression()
    reg1.fit(X, f1_y)
    reg2 = linear_model.LinearRegression()
    reg2.fit(X, f2_y)
    return reg1, reg2

def display_reduction(train, category="phone", reducer=umap.UMAP(), min_sample=1000, display_sample=50):
    df = train.phones_df.groupby(category).filter(lambda x: len(x) > min_sample).groupby(category).sample(n=min_sample)
    X = np.vstack(df["c"])
    y = df.groupby(category).sample(n=display_sample)[category]
    reducer = reducer.fit(X)
    embedding = reducer.transform(np.vstack(df["c"][y.index]))
    color_map = sns.color_palette('Spectral', n_colors=y.nunique())
    sns.scatterplot(embedding[:, 0], embedding[:, 1], hue=y, alpha=0.5, palette=color_map)

    for label, group in df.groupby(category):
        means = np.mean(reducer.transform(np.vstack(group['c'])), axis=0)
        plt.annotate(label, means, ha='center', va='center')
    plt.gca().set_aspect('equal', 'datalim')
    plt.legend()
    plt.show()

data = CPCData(max_files=100)
m = {**{x: 'vowel' for x in TimitData.VOWELS},
     **{x: 'stop' for x in TimitData.STOPS},
     **{x: 'fricative' for x in TimitData.FRICATIVES},
     **{x: 'affricate' for x in TimitData.AFFRICATES},
     **{x: 'nasal' for x in TimitData.AFFRICATES},
     **{x: 'semivowel' for x in TimitData.SEMIVOWELS}}

data.train.phones_df["consonant_type"] = data.train.phones_df["phone"].apply(lambda x: m[x] if x in m else 'non-phone')
data.train.phones_df["vowel"] = data.train.phones_df["phone"].apply(lambda x: m[x] if x in TimitData.VOWELS or x in TimitData.SEMIVOWELS else 'consonant')
data.train.phones_df["vowel"] = data.train.phones_df["phone"].apply(lambda x: m[x] if x in TimitData.VOWELS else 'consonant')

display_reduction(data.train, category='vowel', min_sample=100, display_sample=100)
