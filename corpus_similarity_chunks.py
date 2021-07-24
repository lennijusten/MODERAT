# Description: calculate Spearman correlation coefficient for each chunk between all other chunks
# Author: L. Justen
# Date: July 23, 2021

# References:
# Zwillinger, D. and Kokoska, S. (2000). CRC Standard Probability and Statistics Tables and Formulae.
# Chapman & Hall: New York. 2000. Section 14.7
# Kilgarriff, Adam. (2001). Comparing Corpora. International Journal of Corpus Linguistics. 6. 10.1075/ijcl.6.1.05kil.

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *

chunks_path = '/Data/chunks.pkl'
data_path = '/Data/RP_comments_preprocessed.pkl'

with open(chunks_path, 'rb') as handle:
    chunks = pickle.load(handle)

with open(data_path, 'rb') as handle:
    df = pickle.load(handle)

# vectorizer1 = CountVectorizer(ngram_range=(1, 1), min_df=3)
vectorizer1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3)

corpus = vectorizer1.fit_transform(df['text2'])
corpus_vocab = vectorizer1.get_feature_names()

# Vectorizer for subsets but with complete mapping for corpus vocab
#vectorizer2 = CountVectorizer(ngram_range=(1,1),min_df=3,vocabulary=corpus_vocab)
vectorizer2 = TfidfVectorizer(ngram_range=(1,1),min_df=3,vocabulary=corpus_vocab)

def get_features(comments):
    X = vectorizer2.fit_transform(comments)
    freq_array = X.toarray()
    freq_array_sum = np.sum(freq_array, axis=0)

    return freq_array_sum

def calculate_spear(corpus1, corpus2):
    features1 = get_features(corpus1)
    features2 = get_features(corpus2)

    spear_results = spearmanr(features1, features2)

    return spear_results


col_names = ['C{}'.format(i) for i in range(1, len(chunks) + 1)]
df_rho = pd.DataFrame(columns=[col_names], index=[col_names])
df_p = pd.DataFrame(columns=[col_names], index=[col_names])

for i1 in range(len(chunks)):  # Corpus 1 loop
    col = 'C{}'.format(i1 + 1)  # Training column

    print('Corpus 1 = {}'.format(col))
    corpus1 = chunks[i1]['text2'].values

    p_val = []
    rho = []
    rho.extend([np.nan] * i1)
    p_val.extend([np.nan] * i1)

    print('Looping other chunks')
    for i2 in range(i1, len(chunks)):  # Corpus 2 loop
        corpus2 = chunks[i2]['text2'].values

        rho.append(calculate_spear(corpus1, corpus2)[0])
        p_val.append(calculate_spear(corpus1, corpus2)[1])

    df_rho[col] = rho
    df_p[col] = p_val


mask = np.zeros_like(df_rho)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('Spearman correlation between chunks')
    ax = sns.heatmap(df_rho, cmap="magma_r",mask=mask,square=True)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.invert_yaxis()

df_rho.to_csv('Spearman_rho-CountVect-chunks.csv')

# # Use months for chunks
# months = group_by_month(df)
#
# col_names = [get_date_range(m, include_end=False)[0].strftime('%b-%Y')  for m in months]
# df_rho = pd.DataFrame(columns=[col_names], index=[col_names])
# df_p = pd.DataFrame(columns=[col_names], index=[col_names])
#
# for i1 in range(len(months)):  # Corpus 1 loop
#     col = col_names[i1]
#
#     print('Corpus 1 = {}'.format(col))
#     corpus1 = months[i1]['text2'].values
#
#     p_val = []
#     rho = []
#     rho.extend([np.nan] * i1)
#     p_val.extend([np.nan] * i1)
#
#     print('Looping other chunks')
#     for i2 in range(i1, len(months)):  # Corpus 2 loop
#         corpus2 = months[i2]['text2'].values
#
#         rho.append(calculate_spear(corpus1, corpus2)[0])
#         p_val.append(calculate_spear(corpus1, corpus2)[1])
#
#     print(rho)
#     df_rho[col] = rho
#     df_p[col] = p_val
#
#
# mask = np.zeros_like(df_rho)
# mask[np.triu_indices_from(mask)] = True
# with sns.axes_style("white"):
#     f, ax = plt.subplots(figsize=(7, 5))
#     ax.set_title('Spearman correlation between months')
#     ax = sns.heatmap(df_rho, cmap="magma_r",mask=mask,square=True)
#     ax.set_ylabel('')
#     ax.set_xlabel('')
#     ax.invert_yaxis()
#
# df_rho.to_csv('Spearman_rho-CountVect-chunks.csv')