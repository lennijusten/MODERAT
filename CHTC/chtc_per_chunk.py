import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
import autosklearn
from autosklearn.classification import AutoSklearnClassifier
import argparse
import json
import sys
import shutil
from dateutil.relativedelta import relativedelta
import pickle
import os
import numpy as np

def read_args():
    parser = argparse.ArgumentParser(description='control-strat')
    parser.add_argument("input",
                        type=str,
                        help='name of input file')
    return parser.parse_args()

args = read_args()

with open(args.input, "r") as read_it:
    config = json.load(read_it)


# LOAD DATA
# --------------------------------------------------------------------------------------
chunks_path = config['chunks_path']
train_chunks_path = config['train_chunks_path']
test_chunks_path = config['test_chunks_path']

print("Loading data...")
with open(os.path.basename(chunks_path), 'rb') as handle:
    chunks = pickle.load(handle)

with open(os.path.basename(train_chunks_path), 'rb') as handle:
    train_chunks = pickle.load(handle)

with open(os.path.basename(test_chunks_path), 'rb') as handle:
    test_chunks = pickle.load(handle)

# DEFINE FUNCTIONS
# --------------------------------------------------------------------------------------
def get_date_range(df, include_end=True):
    if include_end:
        return pd.date_range(start=df['Date'].min().replace(day=1).date(),
                           end=df['Date'].max().replace(day=1).date() + relativedelta(months=1),
                           freq='MS')
    else:
        return pd.date_range(start=df['Date'].min().replace(day=1).date(),
                           end=df['Date'].max().replace(day=1).date() + relativedelta(months=1),
                           freq='MS')[0:-1]

def RCR(df):
    return len(df[df['Rejected']==1])/len(df)


# PRE-TRAINING STATISTICS
# --------------------------------------------------------------------------------------
# todo
c_date = []
c_ncomments = []
c_nrejected = []
c_RCR = []
for c in chunks:
    c_date.append(get_date_range(c, include_end=True))
    c_ncomments.append(len(c))
    c_nrejected.append(np.sum(c['Rejected']))
    c_RCR.append(RCR(c))

df_chunks = pd.DataFrame({'DataRange': c_date, 'n_comments': c_ncomments, 'n_rejected': c_nrejected, 'RCR': c_RCR},
                         index=['C{}'.format(i + 1) for i in range(len(chunks))])
df_chunks.to_csv('chunk_stats.csv')


# DEFINE TRAIN/TEST DATA
# --------------------------------------------------------------------------------------
pwd = os.getcwd()
nltk.data.path.append(os.path.join(pwd,'nltk_data'))
german_stop_words = stopwords.words('german')

tf = TfidfVectorizer(stop_words=german_stop_words, max_features=config['tfidf_max_features'],
                     ngram_range=config['tfidf_ngram_range'], min_df=config['tfidf_min_df'])

train_index = config['train_index']
col = 'C{}'.format(train_index + 1)  # Training column
print('Train column = {}'.format(col))

# initialize results dataframe
col_names = ['C{}'.format(i) for i in range(1, len(chunks) + 1)]
df_f1 = pd.DataFrame(columns=col_names, index=col_names)
df_precsion = pd.DataFrame(columns=col_names, index=col_names)
df_recall = pd.DataFrame(columns=col_names, index=col_names)
df_acc = pd.DataFrame(columns=col_names, index=col_names)
df_auc = pd.DataFrame(columns=col_names, index=col_names)

# Pre-define training chunks and fit classifier
X_train = chunks[train_index]['Text 2']
X_train_tf = tf.fit_transform(X_train)
X_train = None

y_train = chunks[train_index]['Rejected']

print('Fitting auto-sklearn to the entire {} chunk'.format(col))
if config['auto-sklearn_include_estimators']['status']:
    automl = AutoSklearnClassifier(
        time_left_for_this_task=43200,
        tmp_folder='auto-sklearn_{}_tmp'.format(col),
        output_folder='auto-sklearn_{}_out'.format(col),
        seed=42,
        memory_limit=config['auto-sklearn_memory_limit'],
        include_estimators=config['auto-sklearn_include_estimators']['estimators'],
        # n_jobs=10,
        metric=autosklearn.metrics.f1
    )
else:
    automl = AutoSklearnClassifier(
        time_left_for_this_task=43200,
        tmp_folder='auto-sklearn_{}_tmp'.format(col),
        output_folder='auto-sklearn_{}_out'.format(col),
        seed=42,
        memory_limit=config['auto-sklearn_memory_limit'],
        # n_jobs=10,
        metric=autosklearn.metrics.f1
    )

automl.fit(X_train_tf, y_train)

Precision = []
Recall = []
F1 = []
acc = []
auc = []
print('Looping through test chunks...')
for test_index in range(len(chunks)):  # Testing loop
    # Evaluation
    if test_index == train_index:  # If testing index == training index, train re-fit classifier on training split of chunk_
        print("Test chunk =  Train chunk... Fitting new 'self' auto-sklearn classifier to train chunk {}".format(col))
        X_train_self = train_chunks[train_index]['Text 2']
        X_train_tf_self = tf.fit_transform(X_train_self)
        X_train_self = None
        y_train_self = train_chunks[train_index]['Rejected']

        X_test = test_chunks[test_index]['Text 2']
        X_test_tf = tf.transform(X_test)
        X_test = None
        y_test = test_chunks[test_index]['Rejected']

        if config['auto-sklearn_include_estimators']['status']:
            automl_self = AutoSklearnClassifier(
                time_left_for_this_task=43200,
                tmp_folder='auto-sklearn_{}self_tmp'.format(col),
                output_folder='auto-sklearn_{}self_out'.format(col),
                seed=42,
                memory_limit=config['auto-sklearn_memory_limit'],
                include_estimators=config['auto-sklearn_include_estimators']['estimators'],
                # n_jobs=10,
                metric=autosklearn.metrics.f1
            )
        else:
            automl_self = AutoSklearnClassifier(
                time_left_for_this_task=43200,
                tmp_folder='auto-sklearn_{}self_tmp'.format(col),
                output_folder='auto-sklearn_{}self_out'.format(col),
                seed=42,
                memory_limit=config['auto-sklearn_memory_limit'],
                # n_jobs=10,
                metric=autosklearn.metrics.f1
            )

        automl_self.fit(X_train_tf_self, y_train_self)

        y_pred = automl_self.predict(X_test_tf)

        test_chunks[train_index]['True'] = y_test
        test_chunks[train_index]['Pred'] = y_pred

        print('Saving self results...')
        test_chunks[train_index].to_csv('{}_self_test_results.csv'.format(col))

        try:
            shutil.make_archive('auto-sklearn_{}self_tmp'.format(col), 'zip', 'auto-sklearn_{}self_tmp'.format(col))
            shutil.make_archive('auto-sklearn_{}self_out'.format(col), 'zip', 'auto-sklearn_{}self_out'.format(col))
        except:
            print('failed to zip tmp folders')
            pass

        with open("{}_Ensemble_self.txt".format(col), "w") as text_file:
            text_file.write(automl_self.show_models())

        with open("{}_sprint_statistics_self.txt".format(col), "w") as text_file:
            text_file.write(automl_self.sprint_statistics())

        df_autosk_self = pd.DataFrame(automl_self.cv_results_)
        df_autosk_self.sort_values(by='rank_test_scores', inplace=True)
        df_autosk_self.to_csv('{}_cv_results_self.csv'.format(col))

        losses_and_configurations_self = [
            (run_value.cost, run_key.config_id)
            for run_key, run_value in automl_self.automl_.runhistory_.data.items()
        ]
        losses_and_configurations_self.sort()
        with open("{}_lowest_loss_config_self.txt".format(col), "w") as text_file:
            text_file.write("Lowest loss: {}".format(losses_and_configurations_self[0][0]) + '\n')
            text_file.write("Best configuration: {}".format(
                automl_self.automl_.runhistory_.ids_config[losses_and_configurations_self[0][1]]))
            text_file.write("\n tfidf params: {}".format(tf))

        test_chunks = None
        train_chunks = None
        automl_self = None
        df_autosk_self = None
        X_train_tf_self = None
        y_train_self = None

    else:
        print('Testing on C{}'.format(test_index))
        X_test = chunks[test_index]['Text 2']
        X_test_tf = tf.transform(X_test)
        y_test = chunks[test_index]['Rejected']

        y_pred = automl.predict(X_test_tf)

        # save predictions from classifier trained on i1 on all other chunks [i2]
        chunks[test_index]['True_train-C{}'.format(col)] = y_test
        chunks[test_index]['Pred_train-C{}'.format(col)] = y_pred

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

print('Saving results...')
with open('{}_results.pkl'.format(col), 'wb') as handle:
    pickle.dump(chunks, handle)

df_precsion.to_csv('precision_{}.csv'.format(col))
df_recall.to_csv('precision_{}.csv'.format(col))
df_f1.to_csv('f1_{}.csv'.format(col))
df_acc.to_csv('acc_{}.csv'.format(col))
df_auc.to_csv('auc_{}.csv'.format(col))

print('Current directory after automl fit')
print(os.listdir(os.curdir))
try:
    shutil.make_archive('auto-sklearn_{}_tmp'.format(col), 'zip', 'auto-sklearn_{}_tmp'.format(col))
    shutil.make_archive('auto-sklearn_{}_out'.format(col), 'zip', 'auto-sklearn_{}_out'.format(col))
except:
    print('failed to zip tmp folders')
    pass

# SAVE RUN RESULTS
# --------------------------------------------------------------------------------------
with open("{}_Ensemble.txt".format(col), "w") as text_file:
    text_file.write(automl.show_models())

with open("{}_sprint_statistics.txt".format(col), "w") as text_file:
    text_file.write(automl.sprint_statistics())

df_autosk = pd.DataFrame(automl.cv_results_)
df_autosk.sort_values(by='rank_test_scores', inplace=True)
df_autosk.to_csv('{}_cv_results.csv'.format(col))

losses_and_configurations = [
    (run_value.cost, run_key.config_id)
    for run_key, run_value in automl.automl_.runhistory_.data.items()
]
losses_and_configurations.sort()
with open("{}_lowest_loss_config.txt".format(col), "w") as text_file:
    text_file.write("Lowest loss: {}".format(losses_and_configurations[0][0]) + '\n')
    text_file.write("Best configuration: {}".format(automl.automl_.runhistory_.ids_config[losses_and_configurations[0][1]]))
    text_file.write("\n tfidf params: {}".format(tf))



