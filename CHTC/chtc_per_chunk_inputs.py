import os
import json
import shutil
import numpy as np

test_name = 'chunking1'
train_chunks = np.arange(0,10,1)
# Set run configuration

dirpath = test_name
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

os.mkdir(dirpath)


for i in train_chunks:
    config = {
        'test_name': test_name,
        'script_name': 'chtc_per_chunk.py',
        'docker_image': 'ljusten/auto-sklearn-lite:latest',
        'train_index': int(i),
        'chunks_path': '/home/ljusten/Moderat/Auto-sklearn/Chunking/Data/chunks.pkl',
        'train_chunks_path': '/home/ljusten/Moderat/Auto-sklearn/Chunking/Data/train_chunks.pkl',
        'test_chunks_path': '/home/ljusten/Moderat/Auto-sklearn/Chunking/Data/test_chunks.pkl',
        'tfidf_max_features': 5000,
        'tfidf_ngram_range': (1, 2),
        'tfidf_min_df': 5,
        'auto-sklearn_time': 43200,
        'auto-sklearn_include_estimators': {'status': False, 'estimators': ['random_forest']},
        'auto-sklearn_memory_limit': 20*1000,
        'request_cpu': 1,
        'request_memory': '26GB',
        'request_disk': '10GB'
    }

    # Write configuration JSON
    fname = "config-{}.json".format(i)
    with open(os.path.join(config['test_name'],fname), "w") as write_file:
        json.dump(config, write_file)

# Write jobs files for multiple runs
with open(os.path.join(dirpath,"jobs.txt"), "w") as text_file:
    for i in range(len(train_chunks)):
        text_file.write('config-{}.json, {}\n'.format(i,i))

# Write shell script
with open(os.path.join(dirpath,"{}.sh".format(test_name)), 'w') as text_file:
    text_file.write('# !/bin/bash\n')
    text_file.write('set -e\n')
    text_file.write('export PATH\n')
    text_file.write('ls\n')
    text_file.write('pwd\n')
    text_file.write('python3 {} $1'.format(config['script_name']))

# Write submit file
with open(os.path.join(config['test_name'],"{}.sub".format(config['test_name'])), 'w') as text_file:
    text_file.write('executable = {}.sh\n'.format(config['test_name']))
    text_file.write('arguments = $(config)\n')

    text_file.write('should_transfer_files = YES\n')
    text_file.write('when_to_transfer_output = ON_EXIT\n')

    text_file.write('transfer_input_files = $(config), {}, {}, {}, {}, /home/ljusten/Moderat/Auto-sklearn/nltk_data\n'
                    .format(config['script_name'], config['chunks_path'], config['train_chunks_path'], config['test_chunks_path']))
    text_file.write('request_cpus = {}\n'.format(config['request_cpu']))
    text_file.write('request_disk = {}\n'.format(config['request_disk']))
    text_file.write('request_memory = {}\n'.format(config['request_memory']))

    text_file.write('error = $(in).err\n')
    text_file.write('log = $(in).log\n'.format(config['test_name']))
    text_file.write('output = $(in).out\n'.format(config['test_name']))

    text_file.write('universe = docker\n')
    text_file.write('docker_image = {}\n'.format(config['docker_image']))

    text_file.write('queue config,in from jobs.txt')
