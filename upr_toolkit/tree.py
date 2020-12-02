from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import nn

from upr_toolkit.timit import TimitData

def generate_tree(sentence_df, do_syllables=True, do_words=True):
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
                if do_syllables:
                    G.add_edge(nucleus, phone)

    if do_words:
        consecutive_words = [(sentence_df['word'] != sentence_df["word"].shift()).cumsum()]
        multi_syllabic_words = sentence_df.loc[nuclei, :].groupby(consecutive_words).filter(lambda x: len(x) > 1).groupby('word')
        for word, group in multi_syllabic_words:
            for x in group.index:
                centre = group[group["stress"] == "1"].index
                if len(centre) == 0:
                    centre = [group[group["stress"] == "0"].first_valid_index()]
                G.add_edge(centre[0], x)

    paths = dict(nx.all_pairs_shortest_path_length(G))
    return G, paths

def get_path_tensor(paths, sentence_max):
    mat = -1*torch.ones((sentence_max, sentence_max))
    offset = min([i for i in paths])
    for i in paths:
        for j in paths[i]:
            mat[i-offset, j-offset] = paths[i][j]
    return mat

def generate_distance_and_c_matrix(phones_df, n_sentences, sentence_max, feature_dim=256, functions=[(torch.mean, {"dim":1})], do_words=True, do_syllables=True):
    distance_tensor = torch.zeros((n_sentences, sentence_max, sentence_max))
    C_tensor = torch.zeros((n_sentences, sentence_max, feature_dim))
    len_tensor = []
    for i, (wav, sentence) in tqdm(enumerate(phones_df.groupby('wav')), total=n_sentences):
        _, paths = generate_tree(sentence, do_syllables, do_words)
        distance_tensor[i, :, :] = get_path_tensor(paths, sentence_max)
        c_values = [torch.from_numpy(x) for x in sentence["c"].values]
        values = torch.hstack([torch.vstack([f(x, **kwargs) for x in c_values])
                            for f, kwargs in functions])
        C_tensor[i, :len(sentence), :] = values
        len_tensor.append(len(sentence))
    return C_tensor, distance_tensor, torch.tensor(len_tensor)


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
    def __init__(self, n_dim=256, B_size=128, sentence_max=66):
        super(TreeProbe, self).__init__()
        self.sentence_max = sentence_max
        self.B = nn.Linear(n_dim, B_size)

    def forward(self, x):
        tree_space = self.B(x)
        tree_space = tree_space.unsqueeze(2).expand(-1, -1, self.sentence_max, -1)
        return torch.sum((tree_space - tree_space.transpose(1, 2)).pow(2), -1)
