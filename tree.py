import os
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from timit import TimitData
from models import Wav2VecData
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
                #if phone == nucleus:
                #    continue
                G.add_edge(nucleus, phone)
    consecutive_words = [(sentence_df['word'] != sentence_df["word"].shift()).cumsum()]
    multi_syllabic_words = sentence_df.loc[nuclei, :].groupby(consecutive_words).filter(lambda x: len(x) > 1).groupby('word')
    noncentres = []
    #for word, group in multi_syllabic_words:
    #    centre = group[group["stress"] == "1"].index
    #    if len(centre) == 0:
    #        centre = [group[group["stress"] == "0"].first_valid_index()]
    #    for x in list(group.index.difference(centre)):
    #        G.add_edge(centre[0], x)
    #        noncentres.append(x)

    ##for nucleus in filter(lambda x: x not in noncentres, nuclei):
    #    G.add_edge(sentence_df.first_valid_index(), nucleus)
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
    C_matrix = np.zeros((n_sentences, sentence_max, 512))
    len_matrix = []
    for i, (wav, sentence) in tqdm(enumerate(phones_df.groupby('wav')), total=n_sentences):
        _, paths = generate_tree(sentence)
        distance_matrix[i, :, :] = get_path_matrix(paths)
        C_matrix[i, :len(sentence), :] = np.concatenate(sentence["c"].values) 
        len_matrix.append(len(sentence))
    return distance_matrix, C_matrix, np.array(len_matrix)

data = Wav2VecData()
N_SENTENCES = data.train.phones_df['wav'].nunique()
SENTENCE_MAX = data.train.phones_df.value_counts('wav').max()
distance_matrix, C_matrix, lengths = generate_distance_and_c_matrix(data.train.phones_df, N_SENTENCES, SENTENCE_MAX)


import torch

dtype = torch.float
device = torch.device("cpu")

#C_tensor = torch.from_numpy(C_matrix).type(torch.float)
#distance_tensor = torch.from_numpy(distance_matrix).type(torch.float)
#lengths = torch.from_numpy(lengths).type(torch.float)

B = torch.rand((512, 32), dtype=torch.float, requires_grad=True)
N_EPOCHS = 40
BATCH_SIZE = 128 
optimizer = torch.optim.Adam([B], lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

def forward(C_batch):
    tree_space = torch.matmul(C_batch, B)
    tree_space = tree_space.unsqueeze(2).expand(-1, -1, SENTENCE_MAX, -1)
    diffs = torch.sum((tree_space - tree_space.transpose(1, 2)).pow(2), -1)
    return diffs

def loss_function(tree_space, distance_batch, sentence_lengths):
    labels_1s = (distance_batch != -1).float()
    loss = torch.sum(torch.abs(tree_space*labels_1s - distance_batch*labels_1s), dim=[1,2]) / sentence_lengths
    loss = torch.sum(loss) / torch.tensor(len(sentence_lengths))
    return loss

for epoch in range(N_EPOCHS):
    for i in range(N_SENTENCES // BATCH_SIZE):
        optimizer.zero_grad()
        min_idx = i*BATCH_SIZE
        max_idx = min_idx + BATCH_SIZE
        C_batch = C_tensor[min_idx:max_idx, :, :] 
        distance_batch = distance_tensor[min_idx:max_idx, :, :]
        tree_space = forward(C_batch)
        loss = loss_function(tree_space, distance_batch, lengths[min_idx:max_idx])
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(loss.item())


N_SENTENCES = data.test.phones_df['wav'].nunique()
SENTENCE_MAX = data.test.phones_df.value_counts('wav').max()
distance_matrix, C_matrix, lengths = generate_distance_and_c_matrix(data.test.phones_df, N_SENTENCES, SENTENCE_MAX)
lengths = torch.from_numpy(lengths)
C_tensor = torch.from_numpy(C_matrix).type(torch.float)
distance_tensor = torch.from_numpy(distance_matrix).type(torch.float)
tree_space = forward(C_tensor)

B = torch.load("B.pt")
labels_1s = (distance_tensor != -1).float()
matching = ((labels_1s*distance_tensor).view(-1) == (torch.round((labels_1s*tree_space).view(-1))//114))[distance_tensor.view(-1) > 0]
print(torch.sum(matching)/float(len(matching)))
output = torch.round(labels_1s*tree_space) // 114
print(output.shape, distance_tensor.shape)
print(labels_1s*output[0, :, :], labels_1s*distance_tensor[0, :, :])
torch.save(B, "B.pt")
