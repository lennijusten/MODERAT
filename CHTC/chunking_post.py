import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from functions import *

# compare two runs
run = '/Users/lenni/PycharmProjects/MODERAT_github/CHTC/months4-lazy50'

save_path = '/Users/lenni/PycharmProjects/MODERAT_github/Figures'

chunks_labels = ['11/2018-02/2019', '03/2019-06/2019', '07/2019-10/2019', '11/2019-02/2020', '03/2020-06/2020']

def prf_matrix(run, metric, save_path=None):
    files_lst = []
    for root, dirs, files in os.walk(run):
        for file in files:
            if file.endswith('{}.csv'.format(metric)):
                files_lst.append(file)

    files_lst.sort()

    cols = ['C{}'.format(i) for i in range(1, len(files_lst)+1)]
    df = pd.DataFrame(columns=cols,index=cols)
    for f, col in zip(files_lst, cols):
        df_tmp = pd.read_csv(os.path.join(run,f), index_col=0)
        df[col] = df_tmp[col]

    plt.figure()
    sns.heatmap(df,annot=True,cmap='crest',yticklabels=chunks_labels,xticklabels=chunks_labels)
    # plt.yticks(rotation=90)
    plt.xticks(rotation=25)
    plt.title('F1-scores')
    plt.xlabel('Train')
    plt.ylabel('Test')
    plt.subplots_adjust(bottom=0.25)
    plt.gca().invert_yaxis()
    if save_path is not None:
        plt.subplots_adjust(bottom=0.18, left=0.25)
        plt.savefig(os.path.join(save_path,'{}_lazy50.jpeg'.format(metric)),dpi=300, bbox_inches="tight")

    plt.show()


prf_matrix(run,'f1',save_path=save_path)