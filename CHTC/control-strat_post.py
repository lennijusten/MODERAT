import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# compare two runs
run1 = '/Users/lenni/PycharmProjects/MODERAT_github/CHTC/lazy-tfidf/control-lazy-tfidf'
run2 = '/Users/lenni/PycharmProjects/MODERAT_github/CHTC/lazy-tfidf/strat-lazy-tfidf'

name1 = os.path.basename(run1)
name2 = os.path.basename(run2)


def compare_monthly_prf(run1, run2, score='F1', save_path=None):
    # choose between F1, Precision, Recall, AUC, Accuracy

    monthly1 = pd.read_csv(os.path.join(run1, 'monthly_results.csv'), index_col=[0])
    monthly2 = pd.read_csv(os.path.join(run2, 'monthly_results.csv'), index_col=[0])

    combined = pd.merge(monthly1, monthly2, on=monthly1.index)
    combined['key_0'] = pd.to_datetime(combined['key_0'])
    combined.set_index('key_0', inplace=True)

    plt.plot(combined.index, combined['{}_x'.format(score)], label=name1)
    plt.plot(combined.index, combined['{}_y'.format(score)], label=name2)
    plt.xlabel('Eval month')
    plt.ylabel('{}-score'.format(score))
    plt.title('{}-score comparison'.format(score))
    plt.xticks(rotation=30)
    plt.legend()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, '{}-comparison.png'.format(score)))

    plt.show()

    return combined


combined = compare_monthly_prf(run1, run2, score='F1',
                               save_path='/Users/lenni/PycharmProjects/MODERAT_github/CHTC/lazy-tfidf')


def comparison_stats(run1, run2, save_path=None):
    with open(os.path.join(run1, 'result_stats.txt'), "r") as tf:
        lines1 = tf.read().split('\n')[:-1]

    with open(os.path.join(run2, 'result_stats.txt'), "r") as tf:
        lines2 = tf.read().split('\n')[:-1]

    stats1 = {
        'F1': float(lines1[0].split(': ')[1]),
        'Precision': float(lines1[1].split(': ')[1]),
        'Recall': float(lines1[2].split(': ')[1]),
        'Accuracy': float(lines1[3].split(': ')[1]),
        'AUC': float(lines1[4].split(': ')[1])
    }

    stats2 = {
        'F1': float(lines2[0].split(': ')[1]),
        'Precision': float(lines2[1].split(': ')[1]),
        'Recall': float(lines2[2].split(': ')[1]),
        'Accuracy': float(lines2[3].split(': ')[1]),
        'AUC': float(lines2[4].split(': ')[1])
    }

    diff = {
        'F1': stats1['F1'] - stats2['F1'],
        'Precision': stats1['Precision'] - stats2['Precision'],
        'Recall': stats1['Recall'] - stats2['Recall'],
        'Accuracy': stats1['Accuracy'] - stats2['Accuracy'],
        'AUC': stats1['AUC'] - stats2['AUC']
    }

    if save_path is not None:
        with open(os.path.join(save_path, 'stats-comparison.txt'), 'w') as text_file:
            text_file.write(name1 + ' minus ' + name2 + '\n')
            text_file.write('F1: {}\n'.format(diff['F1']))
            text_file.write('Precision: {}\n'.format(diff['Precision']))
            text_file.write('Recall: {}\n'.format(diff['Recall']))
            text_file.write('Accuracy: {}\n'.format(diff['Accuracy']))
            text_file.write('AUC: {}\n'.format(diff['AUC']))

    return diff


diff = comparison_stats(run1, run2, save_path='/Users/lenni/PycharmProjects/MODERAT_github/CHTC/lazy-tfidf')

# TODO: Load AutoML models and plot overlapping precision-recall and AUC-ROC curves
