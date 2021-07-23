import os
import json
import shutil

config = {
    'test_name': 'auto-sklearn-json-test',
    'script_name': 'chtc_auto_sklearn.py',
    'docker_image': 'ljusten/auto-sklearn-lite:latest',
    'train_data': 'control', # choose from control and strat
    'eval_path': '/home/ljusten/Moderat/Auto-sklearn/Control-Strat/Data/df_eval_s1-lazy.pkl',
    'control_path': '/home/ljusten/Moderat/Auto-sklearn/Control-Strat/Data/df_control_s1-lazy.pkl',
    'strat_path': '/home/ljusten/Moderat/Auto-sklearn/Control-Strat/Data/df_strat-lazy.pkl',
    'tfifd_max_features': 5000,
    'tfidf_ngram_range': (1, 2),
    'tfidf_min_df': 5,
    'auto-sklearn_time': 300,
    'auto-sklearn_include_estimators': {'status': True, 'estimators': ['random_forest']},
    'request_cpu': 1,
    'request_memory': '15GB',
    'request_disk': '10GB'
}

dirpath = config['test_name']
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

os.mkdir(config['test_name'])

fname = "config-{}.json".format(config['test_name'])
with open(os.path.join(config['test_name'],fname), "w") as write_file:
    json.dump(config, write_file)

with open(os.path.join(config['test_name'],"jobs.txt"), "w") as text_file:
    text_file.write(fname)

with open(os.path.join(config['test_name'],"{}.sh".format(config['test_name'])), 'w') as text_file:
    text_file.write('# !/bin/bash\n')
    text_file.write('set -e\n')
    text_file.write('export PATH\n')
    text_file.write('ls\n')
    text_file.write('pwd\n')
    text_file.write('python3 {} $1'.format(config['script_name']))

with open(os.path.join(config['test_name'],"{}.sub".format(config['test_name'])), 'w') as text_file:
    text_file.write('executable = {}.sh\n'.format(config['test_name']))
    text_file.write('arguments = {}\n'.format(fname))

    text_file.write('should_transfer_files = YES\n')
    text_file.write('when_to_transfer_output = ON_EXIT\n')

    text_file.write('transfer_input_files = {}, {}, {}, {}, {}, /home/ljusten/Moderat/Auto-sklearn/nltk_data\n'
                    .format(config['script_name'], fname, config['eval_path'], config['control_path'], config['strat_path']))
    text_file.write('request_cpus = {}\n'.format(config['request_cpu']))
    text_file.write('request_disk = {}\n'.format(config['request_disk']))
    text_file.write('request_memory = {}\n'.format(config['request_memory']))

    text_file.write('error = {}.err\n'.format(config['test_name']))
    text_file.write('log = {}.log\n'.format(config['test_name']))
    text_file.write('output = {}.out\n'.format(config['test_name']))

    text_file.write('universe = docker\n')
    text_file.write('docker_image = {}\n'.format(config['docker_image']))

    text_file.write('queue')
