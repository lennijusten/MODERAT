import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from TextPreprocessingTransformer import TextPreprocessingTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import autosklearn
from autosklearn.classification import AutoSklearnClassifier
from functions import *

# df = pd.read_csv('comments.csv')
df = pd.read_pickle('/Users/lenni/Downloads/RP_comments_preprocessed.pkl')

# Transform text by removing special characters, removing numbers, converting to lowercase, and Lemmatization
txtpre = TextPreprocessingTransformer()
print('Preprocessing text...')
df['text2'] = txtpre.transform(df['text'].values)

n_chunks = 10
chunks = chunk_by_number(df, n_chunks, method='sequential')

# Split chunks into training (80%) and test splits (20%)
train_chunks = []
test_chunks = []
for c in chunks:
    c_test = c.sample(frac=0.2)
    c_train = c[~c.index.isin(c_test.index)]
    train_chunks.append(c_train)
    test_chunks.append(c_test)

# initialize results dataframe
col_names = ['C{}'.format(i) for i in range(1, len(chunks) + 1)]
df_f1 = pd.DataFrame(columns=[col_names], index=[col_names])
df_precsion = pd.DataFrame(columns=[col_names], index=[col_names])
df_recall = pd.DataFrame(columns=[col_names], index=[col_names])
df_acc = pd.DataFrame(columns=[col_names], index=[col_names])
df_auc = pd.DataFrame(columns=[col_names], index=[col_names])

nltk.download("stopwords")
german_stop_words = stopwords.words('german')

tf = TfidfVectorizer(stop_words=german_stop_words, max_features=3000, ngram_range=(1, 2))

for i1 in range(len(chunks)):  # Training loop
    col = 'C{}'.format(i1 + 1)  # Training column

    # Pre-define training chunks and fit classifier
    X_train = chunks[i1]['text2']
    X_train_tf = tf.fit_transform(X_train)
    y_train = chunks[i1]['rejected']

    automl = AutoSklearnClassifier(
        time_left_for_this_task=43200,
        tmp_folder='auto-sklearn_{}_tmp'.format(col),
        output_folder='auto-sklearn_{}_out'.format(col),
        seed=42,
        memory_limit=None,
        # include_estimators=['random_forest'],
        n_jobs=10,
        metric=autosklearn.metrics.f1
    )

    automl.fit(X_train_tf, y_train)

    Precision = []
    Recall = []
    F1 = []
    acc = []
    auc = []
    for i2 in range(len(chunks)):  # Testing loop
        # Evaluation
        if i2 == i1:  # If testing index == training index, train re-fit classifier on training split of chunk_
            X_train_self = train_chunks[i1]['text2']
            X_train_tf_self = tf.fit_transform(X_train_self)
            y_train_self = train_chunks[i1]['rejected']

            X_test = test_chunks[i1]['text2']
            X_test_tf = tf.fit_transform(X_test)
            y_test = test_chunks[i1]['rejected']

            automl_self = AutoSklearnClassifier(
                time_left_for_this_task=43200,
                tmp_folder='auto-sklearn_{}self_tmp'.format(col),
                output_folder='auto-sklearn_{}self_out'.format(col),
                seed=42,
                memory_limit=None,
                # include_estimators=['random_forest'],
                n_jobs=10,
                metric=autosklearn.metrics.f1
            )

            automl_self.fit(X_train_tf_self, y_train_self)

            y_pred = automl_self.predict(X_test_tf)

            test_chunks[i1]['True'] = y_test
            test_chunks[i1]['Pred'] = y_pred
        else:
            X_test = chunks[i2]['text2']
            X_test_tf = tf.fit_transform(X_test)
            y_test = chunks[i2]['rejected']

            y_pred = automl.predict(X_test_tf)

            # save predictions from classifier trained on i1 on all other chunks [i2]
            chunks[i2]['True_train-C{}'.format(col)] = y_test
            chunks[i2]['Pred_train-C{}'.format(col)] = y_pred

        prf = precision_recall_fscore_support(y_test, y_pred, average="macro")
        Precision.append(prf[0])
        Recall.append(prf[1])
        F1.append(prf[2])
        acc.append(accuracy_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred))

    df_precsion[col] = Precision
    df_recall[col] = Recall
    df_f1[col] = F1
    df_acc[col] = acc
    df_auc[col] = auc
