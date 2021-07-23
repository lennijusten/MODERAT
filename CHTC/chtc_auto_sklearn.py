import pickle
import numpy as np
import pandas as pd
import os
from dateutil.relativedelta import relativedelta
import autosklearn
from autosklearn.classification import AutoSklearnClassifier
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, precision_recall_fscore_support, plot_confusion_matrix,plot_precision_recall_curve, plot_roc_curve
import argparse
import json
import sys

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
eval_path = config['eval_path']
control_path = config['control_path']
strat_path = config['strat_path']

print("Loading data...")
with open(os.path.basename(eval_path), 'rb') as handle:
    df_eval_s1 = pickle.load(handle)

with open(os.path.basename(control_path), 'rb') as handle:
    df_control_s1 = pickle.load(handle)

with open(os.path.basename(strat_path), 'rb') as handle:
    df_strat = pickle.load(handle)


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


def group_by_month(df):
    g = df.groupby(pd.Grouper(key='Date', freq='M'))

    # groups to a list of dataframes with list comprehension
    return [group for _, group in g]

# PRE-TRAINING STATISTICS
# --------------------------------------------------------------------------------------
# p_control = len(df_control_s1[df_control_s1.index.isin(df_eval.index)]) / len(df_control_s1) * 100
m_control = get_date_range(df_control_s1)
m_strat = get_date_range(df_strat)
m_eval = get_date_range(df_eval_s1)

print('Writing dataset statistics...')
with open("dataset_stats.txt", "w") as text_file:
    text_file.write("control dataset: {}\n".format(control_path))
    text_file.write("length of control dataset: {}\n".format(len(df_control_s1)))
    text_file.write("n rejected in control dataset: {}\n".format(np.sum(df_control_s1['Rejected'])))
    text_file.write("RCR control dataset: {}\n\n".format(RCR(df_control_s1)))

    text_file.write("stratified dataset: {}\n".format(strat_path))
    text_file.write("length of stratified dataset: {}\n".format(len(df_strat)))
    text_file.write("n rejected in stratified dataset: {}\n".format(np.sum(df_strat['Rejected'])))
    text_file.write("RCR stratified dataset: {}\n\n".format(RCR(df_strat)))

    text_file.write("eval dataset: {}\n".format(control_path))
    text_file.write("length of eval dataset: {}\n".format(len(df_eval_s1)))
    text_file.write("n rejected in eval dataset: {}\n".format(np.sum(df_eval_s1['Rejected'])))
    text_file.write("RCR eval dataset: {}\n\n".format(RCR(df_eval_s1)))

    # text_file.write("Proportion of control dataset in evaluation period: {}\n".format(p_control))
    text_file.write("control dataset daterange: {}\n".format(m_control))
    text_file.write("stratified dataset daterange: {}\n".format(m_strat))
    text_file.write("eval dataset daterange: {}\n".format(m_eval))


RCR_control_bymonth = [RCR(month) for month in group_by_month(df_control_s1)]
n_control_bymonth = [len(month) for month in group_by_month(df_control_s1)]
n_control_rejected_bymonth = [np.sum(month['Rejected']) for month in group_by_month(df_control_s1)]

RCR_strat_bymonth = [RCR(month) for month in group_by_month(df_strat)]
n_strat_bymonth = [len(month) for month in group_by_month(df_strat)]
n_strat_rejected_bymonth = [np.sum(month['Rejected']) for month in group_by_month(df_strat)]

RCR_eval_bymonth = [RCR(month) for month in group_by_month(df_eval_s1)]
n_eval_bymonth = [len(month) for month in group_by_month(df_eval_s1)]
n_eval_rejected_bymonth = [np.sum(month['Rejected']) for month in group_by_month(df_eval_s1)]

control_bymonth = pd.DataFrame({'comment':n_control_bymonth,'rejected':n_control_rejected_bymonth,'RCR':RCR_control_bymonth},index=m_control[:-1])
strat_bymonth = pd.DataFrame({'comment':n_strat_bymonth,'rejected':n_strat_rejected_bymonth,'RCR':RCR_strat_bymonth},index=m_strat[:-1])
eval_bymonth = pd.DataFrame({'comment':n_eval_bymonth,'rejected':n_eval_rejected_bymonth,'RCR':RCR_eval_bymonth},index=m_eval[:-1])

control_bymonth.to_csv('control_s1_bymonth.csv')
strat_bymonth.to_csv('strat_bymonth.csv')
eval_bymonth.to_csv('eval_s1_bymonth.csv')

# DEFINE TRAIN/TEST DATA
# --------------------------------------------------------------------------------------
print('Defining training data and fitting transformer...')
if config['train_data'] == 'control':
    X_train = df_control_s1['Text 2'].values
    y_train = df_control_s1['Rejected'].values
elif config['train_data'] == 'strat':
    X_train = df_strat['Text 2'].values
    y_train = df_strat['Rejected'].values
else:
    print("wrong training data value given. Choose between 'control' and 'strat' in the config file")
    sys.exit()

X_test = df_eval_s1['Text 2'].values
y_test = df_eval_s1['Rejected'].values

pwd = os.getcwd()
nltk.data.path.append(os.path.join(pwd,'nltk_data'))
nltk.download("stopwords")
german_stop_words = stopwords.words('german')

# Autosklearn can't optimize n_features and ngram_range this way
# Possible solution: build preprocessing class from autosklearn and add manually
tf = TfidfVectorizer(stop_words=german_stop_words, max_features=config['tfidf_max_features'],
                     ngram_range=config['tfidf_ngram_range'], min_df=config['tfidf_min_df'])

X_train_tf = tf.fit_transform(X_train)
X_test_tf = tf.fit_transform(X_test)

# FIT AUTO-SKLEARN
# --------------------------------------------------------------------------------------

print('Fitting auto-sklearn')
if config['auto-sklearn_include_estimators']['status']:
    automl = AutoSklearnClassifier(
        time_left_for_this_task=config['auto-sklearn_time'],
        tmp_folder='{}_tmp'.format(config['test_name']),
        output_folder='{}_out'.format(config['test_name']),
        seed=42,
        memory_limit=None,
        include_estimators=config['auto-sklearn_include_estimators']['estimators'],
        # n_jobs=8,
        metric=autosklearn.metrics.f1
    )
else:
    automl = AutoSklearnClassifier(
        time_left_for_this_task=config['auto-sklearn_time'],
        tmp_folder='{}_tmp'.format(config['test_name']),
        output_folder='{}_out'.format(config['test_name']),
        seed=42,
        memory_limit=None,
        # n_jobs=8,
        metric=autosklearn.metrics.f1
    )


automl.fit(X_train_tf, y_train)

print('Auto-sklearn fit done. Predicting test data...')
y_pred = automl.predict(X_test_tf)

df_eval_s1['True'] = y_test
df_eval_s1['Pred'] = y_pred

df_eval_s1.to_csv('results.csv')

# SAVE RUN PERFORMANCE
print('Creating pretty performance files...')
prf = precision_recall_fscore_support(df_eval_s1['True'], df_eval_s1['Pred'], average="macro")
print(prf)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

with open("result_stats.txt", "w") as text_file:
    text_file.write('F1-score: {}\n'.format(prf[2]))
    text_file.write('Precision: {}\n'.format(prf[0]))
    text_file.write('Recall: {}\n'.format(prf[1]))
    text_file.write('Accuracy: {}\n'.format(acc))
    text_file.write('AUC: {}\n'.format(auc))

months = group_by_month(df_eval_s1)
f1_m = [precision_recall_fscore_support(df['True'], df['Pred'], average="macro")[2] for df in months]
precision_m = [precision_recall_fscore_support(df['True'], df['Pred'], average="macro")[0] for df in months]
recall_m = [precision_recall_fscore_support(df['True'], df['Pred'], average="macro")[1] for df in months]
acc_m = [accuracy_score(df['True'], df['Pred']) for df in months]
auc_m = [roc_auc_score(df['True'], df['Pred']) for df in months]

df_monthly_perf = pd.DataFrame({'F1':f1_m,'Precision':precision_m,'Recall':recall_m,'Accuracy':acc_m,'AUC':auc_m},index=m_eval[:-1])
df_monthly_perf.to_csv('monthly_results.csv')

fig, ax = plt.subplots()
plot_roc_curve(automl, X_test_tf, y_test, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)
plt.savefig('ROC_curve.png')

fig, ax = plt.subplots()
plot_precision_recall_curve(automl, X_test_tf, y_test,ax=ax)
plt.savefig('Precision-Recall_curve.png')

fig, ax = plt.subplots()
plot_confusion_matrix(automl, X_test_tf, y_test)
plt.savefig('confusion-matrix.png')


def plot_monthly_perf(y, metric='F1-score'):
    x = np.arange(0, len(y), 1)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    fig, ax = plt.subplots()
    ax.set_title('Per-month {} of evaluation dataset'.format(metric))
    ax.set_xlabel('Months since start of evaluation data split')
    ax.set_ylabel('{}'.format(metric))
    ax.plot(y, 'o')
    ax.plot(x, p(x), "r--")
    plt.savefig('monthly_F1.png')
    return fig


# SAVE RUN RESULTS
# --------------------------------------------------------------------------------------
with open("Ensemble.txt", "w") as text_file:
    text_file.write(automl.show_models())

with open("sprint_statistics.txt", "w") as text_file:
    text_file.write(automl.sprint_statistics())

df_autosk = pd.DataFrame(automl.cv_results_)
df_autosk.sort_values(by='rank_test_scores', inplace=True)
df_autosk.to_csv('cv_results.csv')

losses_and_configurations = [
    (run_value.cost, run_key.config_id)
    for run_key, run_value in automl.automl_.runhistory_.data.items()
]
losses_and_configurations.sort()
with open("lowest_loss_config.txt", "w") as text_file:
    text_file.write("Lowest loss: {}".format(losses_and_configurations[0][0]) + '\n')
    text_file.write("Best configuration: {}".format(automl.automl_.runhistory_.ids_config[losses_and_configurations[0][1]]))
    text_file.write("\n tfidf params: {}".format(tf))