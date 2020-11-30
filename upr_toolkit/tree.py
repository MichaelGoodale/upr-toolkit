import os
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from torch import nn

from timit import TimitData
from models import Wav2VecData, CPCData, RandomData, Wav2Vec2Data
import networkx as nx

def generate_tree(sentence_df):
    G = nx.Graph()

    syllable_groups = []
    syllable_group = [] 

    word_groups = []
    word_group = []
    for i, r in sentence_df.iterrows():
        G.add_node(i, **r.to_dict())

        if r['boundary'] in ["1", "2"]:
            if syllable_group != []:
                syllable_groups.append(syllable_group)
            syllable_group = []
        syllable_group.append(i)

    nuclei = []
    for syllable_group in syllable_groups:
        y = sentence_df.loc[syllable_group, :]
        y = y[y["stress"].str.match('1|2|0')]
        if len(y) > 0:
            nucleus = y.index[0]
            nuclei.append(nucleus)
            for phone in syllable_group:
                if phone == nucleus:
                    continue
                G.add_edge(nucleus, phone)
    consecutive_words = [(sentence_df['word'] != sentence_df["word"].shift()).cumsum()]
    multi_syllabic_words = sentence_df.loc[nuclei, :].groupby(consecutive_words).filter(lambda x: len(x) > 1).groupby('word')
    noncentres = []
    for word, group in multi_syllabic_words:
        centre = group[group["stress"] == "1"].index
        if len(centre) == 0:
            centre = [group[group["stress"] == "0"].first_valid_index()]
        for x in list(group.index.difference(centre)):
            G.add_edge(centre[0], x)
            noncentres.append(x)

    for nucleus in filter(lambda x: x not in noncentres, nuclei):
        G.add_edge(sentence_df.first_valid_index(), nucleus)
    paths = dict(nx.all_pairs_shortest_path_length(G))
    return G, paths

def get_path_matrix(paths):
    mat = -1*np.ones((SENTENCE_MAX, SENTENCE_MAX))
    offset = min([i for i in paths])
    for i in paths:
        for j in paths[i]:
            mat[i-offset, j-offset] = paths[i][j]
    return mat

def generate_distance_and_c_matrix(phones_df, n_sentences, sentence_max):
    distance_matrix = np.zeros((n_sentences, sentence_max, sentence_max))
    C_matrix = np.zeros((n_sentences, sentence_max, 304))
    len_matrix = []
    for i, (wav, sentence) in tqdm(enumerate(phones_df.groupby('wav')), total=n_sentences):
        _, paths = generate_tree(sentence)
        distance_matrix[i, :, :] = get_path_matrix(paths)
        values = np.hstack([np.vstack([np.mean(x, axis=1) for x in sentence["c"].values]),
                            np.vstack([np.var(x, axis=1) for x in sentence["c"].values]),
                            np.vstack([np.median(x, axis=1) for x in sentence["c"].values]),
                            np.vstack([x.shape[1] for x in sentence["c"].values])])
        C_matrix[i, :len(sentence), :] = values
        len_matrix.append(len(sentence))
    return distance_matrix, C_matrix, np.array(len_matrix)


def tree_loss(output, true_output, sentence_lengths):
    labels_1s = (true_output != -1).float()
    loss = torch.sum(torch.abs(output*labels_1s - true_output*labels_1s), dim=[1,2]) / sentence_lengths
    loss = torch.sum(loss) / torch.tensor(len(sentence_lengths))
    return loss

def per_unit_acc(output, ground_truth, sentence_lengths):
    output = torch.round(output)
    labels_1s = (ground_truth != -1).float()
    matching = ((labels_1s*ground_truth).view(-1) == ((labels_1s*output).view(-1)))
    return torch.sum(matching) / float(len(matching))

class TreeProbe(nn.Module):
    def __init__(self, n_dim=304, B_size=128, sentence_max=66):
        super(TreeProbe, self).__init__()
        self.sentence_max = sentence_max
        self.B = nn.Linear(n_dim, B_size, dtype=torch.float, requires_grad=True)

    def forward(self, x):
        tree_space = self.B(C_batch)
        tree_space = tree_space.unsqueeze(2).expand(-1, -1, SENTENCE_MAX, -1)
        return torch.sum((tree_space - tree_space.transpose(1, 2)).pow(2), -1)
