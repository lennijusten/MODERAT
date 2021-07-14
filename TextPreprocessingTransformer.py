from tqdm import tqdm
import re
import nltk
from sklearn.base import TransformerMixin

class TextPreprocessingTransformer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        documents = []

        for sen in tqdm(range(0, len(X))):
            # Remove all the special characters
            document = re.sub(r'\W', ' ', str(X[sen]))

            # Remove numbers
            document = re.sub(r'[0-9]', ' ', document)

            # remove all single characters
            document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

            # Remove single characters from the start
            document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

            # Substituting multiple spaces with single space
            document = re.sub(r'\s+', ' ', document, flags=re.I)

            # Removing prefixed 'b'
            document = re.sub(r'^b\s+', '', document)

            # Converting to Lowercase
            document = document.lower()

            # Lemmatization
            snowball = nltk.SnowballStemmer(language= "german")

            document = document.split()

            document = [snowball.stem(word) for word in document]
            document = ' '.join(document)

            documents.append(document)

        return documents