import os
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from timit import TimitData
from models import Wav2VecData
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



data = Wav2VecData()

idx = data.train.phones_df["phone"].isin(TimitData.VOWELS)
phones = data.train.phones_df[idx]
X = np.vstack(phones["c"])
f1_y = phones["f1"]
f2_y = phones["f2"]

idx_test = data.test.phones_df["phone"].isin(TimitData.VOWELS)
phones_test = data.test.phones_df[idx_test]
X_test = np.vstack(phones_test["c"])
f1_y_test = phones_test["f1"]
f2_y_test = phones_test["f2"]

from sklearn import linear_model
from sklearn import svm
reg1 = linear_model.LinearRegression()
reg1.fit(X, f1_y)
print(reg1.score(X, f1_y))
print(reg1.score(X_test, f1_y_test))
reg2 = linear_model.LinearRegression()
reg2.fit(X, f2_y)
print(reg2.score(X, f2_y))
print(reg2.score(X_test, f2_y_test))
#
##X_test = np.vstack(phones_test["c"])
#X_test = np.column_stack((reg2.predict(X_test), reg1.predict(X_test)))
#
#h = 10
#plt.ylim(200, 1100)
#plt.xlim(500, 3100)
#x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
#y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
#xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                     np.arange(y_min, y_max, h))
#
#for vowel in ["iy", "aa", "ae", "eh"]:
#    Y_test = phones_test["phone"].apply(lambda x: int(x == vowel))
#    clf = svm.SVC(kernel = 'rbf')
#    clf.fit(X_test, Y_test)
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#    plt.contour(xx, yy, Z, label=vowel)
#    idx = phones_test["phone"] == vowel
#    plt.scatter(f2_y_test[idx], f1_y_test[idx], label=vowel,  alpha=0.5)
#plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()
#plt.xlabel("F2")
#plt.ylabel("F1")
#plt.legend()
#plt.show()

phones = data.train.phones_df.groupby("phone").filter(lambda x: len(x) > 1000).groupby("phone").sample(n=1000)
X = np.vstack(phones["c"])
y = phones["phone"].sample(n=1000)
#pca = PCA(n_components=50).fit_transform(X)
#embedding = TSNE(n_components=2).fit_transform(pca)
umapper = umap.UMAP().fit(X)
embedding = umapper.transform(np.vstack(phones["c"][y.index]))

color_map = sns.color_palette('Spectral', n_colors=y.nunique())
sns.scatterplot(embedding[:, 0], embedding[:, 1], hue=y, alpha=0.5, palette=color_map)
for phone, group in phones.groupby("phone"):
    means = np.mean(umapper.transform(np.vstack(group['c'])), axis=0)
    plt.annotate(phone, means, ha='center', va='center')
plt.gca().set_aspect('equal', 'datalim')
plt.legend()
plt.show()
