import pandas as pd
from functions import *
from TextPreprocessingTransformer import TextPreprocessingTransformer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
import autosklearn
from autosklearn.classification import AutoSklearnClassifier

# Read csv file with columns [text,date,rejected]
df = pd.read_csv('comments.csv')

# Transform text by removing special characters, removing numbers, converting to lowercase, and Lemmatization
txtpre = TextPreprocessingTransformer()
print('Preprocessing text...')
df['text2'] = txtpre.transform(df['text'].values)

# Set data
eval_months = 8

date_range = get_date_range(df)  # Get pandas datetime index
n_months = len(date_range)-1

# Split dataset into control, stratified, and evaluation by time index
m_eval, df_eval = last_months(df,eval_months,group=False)
m_strat, df_strat = first_months(df,n_months-eval_months,group=False)
m_control, df_control = time_interval(df,1,n_months,group=False)

# Undersample control dataset to be same length as the time stratified dataset
df_control_s1 = random_undersampling(df_control,len(df_strat),seed=42)

# Update eval dataset to exclude comments in the control dataset
df_eval_s1 = df_eval[~df_eval.index.isin(df_control_s1.index)]


# Explicitly define testing and training data
X_train_control = df_control_s1['text2'].values
y_train_control = df_control_s1['rejected'].values

X_train_strat = df_strat['text2'].values
y_train_strat = df_strat['rejected'].values

X_test = df_eval_s1['text2'].values
y_test = df_eval_s1['rejected'].values

# Transform text2 in tfidf vector representations
nltk.download("stopwords")
german_stop_words = stopwords.words('german')

tf = TfidfVectorizer(stop_words=german_stop_words, max_features=3000, ngram_range=(1,2))

print('Converting text to tfidf vectors...')
X_train_control_tf = tf.fit_transform(X_train_control)
X_train_strat_tf = tf.fit_transform(X_train_strat)

X_test_tf = tf.fit_transform(X_test)


# Fit auto-sklearn to control dataset
print("Fitting control split with autosklearn...")
automl_control = AutoSklearnClassifier(
    time_left_for_this_task=86400,
    tmp_folder='auto-sklearn_control_tmp',
    output_folder='auto-sklearn_control_out',
    seed=42,
    memory_limit=None,
    # include_estimators=['random_forest'],
    n_jobs=10,
    metric=autosklearn.metrics.f1
)

automl_control.fit(X_train_control_tf, y_train_control)

y_pred_control = automl_control.predict(X_test_tf)

df_eval['control_preds'] = y_pred_control

prf_control = precision_recall_fscore_support(y_test, y_pred_control, average="macro")

print("Control split performance:")
print("Precision={}".format(prf_control[0]))
print("Recall={}".format(prf_control[1]))
print("F1-score={}".format(prf_control[2]))


# Fit auto-sklearn to stratified dataset
print("Fitting stratified split with autosklearn...")
automl_strat = AutoSklearnClassifier(
    time_left_for_this_task=86400,
    tmp_folder='auto-sklearn_strat_tmp',
    output_folder='auto-sklearn_strat_out',
    seed=42,
    memory_limit=None,
    # include_estimators=['random_forest'],
    n_jobs=10,
    metric=autosklearn.metrics.f1
)

automl_strat.fit(X_train_strat_tf, y_train_strat)

y_pred_strat = automl_strat.predict(X_test_tf)

df_eval['strat_preds'] = y_pred_strat

prf_strat = precision_recall_fscore_support(y_test, y_pred_strat, average="macro")

print("Stratified split performance:")
print("Precision={}".format(prf_strat[0]))
print("Recall={}".format(prf_strat[1]))
print("F1-score={}".format(prf_strat[2]))

df_eval[['text','date','rejected','control_preds','strat_preds']].to_csv('results.csv')
