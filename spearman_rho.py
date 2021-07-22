from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class Similarity(object):

    def __init__(self):
        # pass corpus 1 and corpus 2 as arrays of transformed strings

        self.ngram_range = (1,1)
        self.text_feature = 'word'
        self.min_df = 1
        self.vectorizer = CountVectorizer(analyzer=self.text_feature, ngram_range=self.ngram_range,vocabulary=None)

    def get_features(self, comments):

        X = self.vectorizer.fit_transform(comments)
        freq_array = X.toarray()
        freq_array_sum = np.sum(freq_array, axis=0)

        return freq_array_sum

    def calculate(self, corpus1, corpus2):

        features1 = self.get_features(corpus1)
        features2 = self.get_features(corpus2)

        spear_results = spearmanr(features1, features2)[0]

        return spear_results

