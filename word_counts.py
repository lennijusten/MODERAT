# Description: Return words that emerge for the first time in the evaluation time period of the control-stratified test.
# Author: L. Justen
# Date: July 21, 2021

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from functions import *

# Load data with already text (pre-processed text recommended)
control_path = 'Data/df_control_s1-lazy.pkl'
strat_path = 'Data/df_strat-lazy.pkl'
eval_path = 'Data/df_eval.pkl'

with open(eval_path, 'rb') as handle:
    df_eval = pickle.load(handle)

with open(control_path, 'rb') as handle:
    df_control_s1 = pickle.load(handle)

with open(strat_path, 'rb') as handle:
    df_strat = pickle.load(handle)


vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=3)


def per_comment_wc(df, vectorizer=vectorizer):
    # return dataframe with rows:commentID, cols:unique words in corp
    feature_count = vectorizer.fit_transform(df['text2'])  # Fit count vectorizer
    features = vectorizer.get_feature_names()  # get unique words in corpa
    return pd.DataFrame(feature_count.toarray(), columns=features, index=df.index)


def get_new_words(A, B, rejected=False, return_top=20):
    # Return top n words in A that are not in B
    # To return top n words in A rejected comments not in B rejected comments use rejected=True

    if rejected:
        a_words = per_comment_wc(A[A['rejected']==1])
        b_words = per_comment_wc(B[B['rejected']==1])
    else:
        a_words = per_comment_wc(A)
        b_words = per_comment_wc(B)

    diff = a_words.columns.difference(b_words.columns)  # Get words of A not in B
    print("Number of words in A not in B = {}".format(len(diff)))

    # save memory
    b_words = None

    unique_words = a_words[a_words.columns.intersection(diff.values)]

    a_words = None

    freq_unique = unique_words.sum(min_count=1)
    freq_unique.sort_values(axis=0, inplace=True, ascending=False)
    return freq_unique.head(return_top)


new_words = get_new_words(df_strat,df_eval, rejected=False)

new_words.to_csv('words_in_A_not_in_B.csv')











