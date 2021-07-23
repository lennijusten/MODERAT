# Description: calculate Spearman correlation coefficient with associated p-value for the control, stratified,
# and evaluation dataset
# Author: L. Justen
# Date: July 21, 2021

# References:
# Zwillinger, D. and Kokoska, S. (2000). CRC Standard Probability and Statistics Tables and Formulae.
# Chapman & Hall: New York. 2000. Section 14.7
# Kilgarriff, Adam. (2001). Comparing Corpora. International Journal of Corpus Linguistics. 6. 10.1075/ijcl.6.1.05kil.

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.stats import spearmanr

# Load data with already text (pre-processed text recommended)
data_path = 'Data/comment.pkl'  # comments over the entire corpus
control_path = 'Data/df_control_s1-lazy.pkl'
strat_path = 'Data/df_strat-lazy.pkl'
eval_path = 'Data/df_eval_s1-lazy.pkl'

with open(data_path, 'rb') as handle:
    df = pickle.load(handle)

with open(eval_path, 'rb') as handle:
    df_eval_s1 = pickle.load(handle)

with open(control_path, 'rb') as handle:
    df_control_s1 = pickle.load(handle)

with open(strat_path, 'rb') as handle:
    df_strat = pickle.load(handle)

# Get indexed vocabulary for the entire corpus (words occuring at least three times)
vectorizer1 = CountVectorizer(ngram_range=(1, 1), min_df=3)
# vectorizer1 = TfidfVectorizer(ngram_range=(1, 1), min_df=3)

corpus = vectorizer1.fit_transform(df['text2'])
corpus_vocab = vectorizer1.get_feature_names()

# Vectorizer for subsets but with complete mapping for corpus vocab
vectorizer2 = CountVectorizer(ngram_range=(1,1),min_df=3,vocabulary=corpus_vocab)
# vectorizer2 = TfidfVectorizer(ngram_range=(1,1),min_df=3,vocabulary=corpus_vocab)

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

# Returns spearman rho and associated p-value
rho_eval_control = calculate_spear(df_eval_s1['text2'].values, df_control_s1['text2'].values)
rho_eval_strat = calculate_spear(df_eval_s1['text2'].values, df_strat['text2'].values)
rho_strat_control = calculate_spear(df_control_s1['text2'].values, df_eval_s1['text2'].values)