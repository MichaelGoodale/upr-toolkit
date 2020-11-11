import os
from itertools import product
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from timit import TimitData
from models import Wav2VecData, CPCData, VQWav2VecData
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

def conditional_probability_matrix(train, vq_column=0, category="phone"):
    units = np.round(np.vstack(train.phones_df["c"]))[:, vq_column].astype(int)
    probability_of_unit = {unique: count/len(units) for unique, count in zip(*np.unique(units, return_counts=True))}
    probability_of_phone_and_unit = {(unit, cat): 0 for unit, cat in product(probability_of_unit.keys(), train.phones_df[category].unique())}
    for i, row in train.phones_df.iterrows():
        cat = row[category]
        unit = units[i]
        probability_of_phone_and_unit[(unit, cat)] += 1
    probability_of_phone_and_unit = {k: v/len(train.phones_df) for k, v in probability_of_phone_and_unit.items()}
    probability_of_phone_given_unit = {(cat, unit): v/probability_of_unit[unit] for (unit, cat), v in probability_of_phone_and_unit.items()}

    cat_list = train.phones_df[category].value_counts()
    cat_list = list(cat_list[cat_list > 25].index)
    prob_mat = np.array([[probability_of_phone_given_unit[(cat, unit)] for unit in probability_of_unit.keys()] for cat in cat_list])
    #Find the highest phone for each given unit, then sort so that we go from the highest prob for phone 0, phone 1, etc...
    keys = [i for _, _, i in sorted([(a, prob_mat[a, i], i) for i, a in enumerate(np.argmax(prob_mat, 0))])]
    prob_mat = np.clip(prob_mat, 0., 0.5) #Saturate colours @ 0.5
    plt.matshow(prob_mat[:, keys])
    plt.yticks(np.arange(len(cat_list)), cat_list)
    plt.show()

data = VQWav2VecData()
conditional_probability_matrix(data.train, vq_column=0)
