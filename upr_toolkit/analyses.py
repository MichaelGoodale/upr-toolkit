import os
from itertools import product
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from upr_toolkit.timit import TimitData
from upr_toolkit.models import Wav2VecData, CPCData, VQWav2VecData
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import linear_model
import torchaudio

def get_formant_regression(train):
    X, f1_y, f2_y = get_formant_data(train)
    reg1 = linear_model.LinearRegression()
    reg1.fit(X, f1_y)
    reg2 = linear_model.LinearRegression()
    reg2.fit(X, f2_y)
    return reg1, reg2

def get_formant_data(train):
    idx = train.phones_df["phone"].isin(TimitData.VOWELS)
    phones = train.phones_df[idx]
    X = np.hstack(phones["c"]).T
    f1_y = np.hstack(phones["f1"])
    f2_y = np.hstack(phones["f2"])
    return X, f1_y, f2_y

def compare_formants(data):
    reg1, reg2 = get_formant_regression(data.train)
    X, f1_y, f2_y = get_formant_data(data.test)
    return reg1.score(X, f1_y), reg2.score(X, f2_y)

def display_reduction(train, category="phone", reducer=umap.UMAP(), min_sample=1000, display_sample=50):
    df = train.phones_df.groupby(category).filter(lambda x: len(x) > min_sample).groupby(category).sample(n=min_sample)
    X = np.vstack([np.mean(x, axis=2) for x in df["c"]])
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

def conditional_probability_matrix(train, vq_column=0, category="phone"):
    units = np.round(np.hstack(train.phones_df["c"]))[vq_column, :].astype(int)
    probability_of_unit = {unique: count/len(units) for unique, count in zip(*np.unique(units, return_counts=True))}

    probability_of_phone_and_unit = {(unit, cat): 0 for unit, cat in product(probability_of_unit.keys(), train.phones_df[category].unique())}
    for i, row in train.phones_df.iterrows():
        cat = row[category]
        for unit, count in zip(*np.unique(row["c"][vq_column, :], return_counts=True)):
            probability_of_phone_and_unit[(unit, cat)] += count
    probability_of_phone_and_unit = {k: v/len(units) for k, v in probability_of_phone_and_unit.items()}

    probability_of_phone_given_unit = {(cat, unit): v/probability_of_unit[unit] for (unit, cat), v in probability_of_phone_and_unit.items()}

    cat_list = train.phones_df[category].value_counts()
    cat_list = list(cat_list[cat_list > 25].index)
    prob_mat = np.array([[probability_of_phone_given_unit[(cat, unit)] for unit in probability_of_unit.keys()] for cat in cat_list])
    #Find the highest phone for each given unit, then sort so that we go from the highest prob for phone 0, phone 1, etc...
    keys = [i for _, _, i in sorted([(a, -prob_mat[a, i], i) for i, a in enumerate(np.argmax(prob_mat, 0))])]
    prob_mat = np.clip(prob_mat, 0., 0.5) #Saturate colours @ 0.5
    plt.matshow(prob_mat[:, keys])
    plt.yticks(np.arange(len(cat_list)), cat_list)
    plt.show()

def spectrogram_and_encodings(train, wav='timit/TIMIT/train/dr4/msrg0/sa1.wav'):
    sentence = train.phones_df[train.phones_df["wav"] == wav]
    signal, _ = torchaudio.load(wav)
    fig, axs = plt.subplots(2)
    signal = signal.numpy()[0]
    signal = signal[:int(0.25*len(signal))]
    axs[0].specgram(signal)

    axs[1].plot(signal)
    axs[1].set_xlim(0, len(signal))
    bottom, top = axs[1].get_ylim()
    c_pos = 0
    for i, row in sentence.iterrows():
        if i != 0: 
            axs[1].axvline(x=row["start"], c="red")
        axs[1].text( (row['end'] - row['start']) / 2 + row['start'],
                top - 0.005,
                row["phone"],
                verticalalignment="top",
                horizontalalignment="center")
        c = row["c"]
        c_offset = (row['end']-row['start'])/c.shape[-1]
        for unit in c.T:
            axs[1].text(c_pos, bottom + 0.005, unit, rotation="vertical", fontsize='x-small')
            c_pos += c_offset

    plt.show()
