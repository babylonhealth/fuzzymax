# Copyright 2018 Babylon Partners. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This source code is derived from SentEval source code.
# SentEval Copyright (c) 2017-present, Facebook, Inc.
# ==============================================================================

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import itertools

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'


sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
import utils
from similarity import get_similarity_by_name
from evaluation.constants import *

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def prepare(params, samples):
    word_vec_path = utils.get_word_vec_path_by_name(params.word_vec_name)
    word_count_path = params.word_count_path
    norm = params.norm
    params.wvec_dim = 300

    _, params.word2id = utils.create_dictionary(samples)
    params.word_vec = utils.get_wordvec(word_vec_path,
                                        params.word2id,
                                        norm=norm,
                                        path_to_counts=word_count_path)
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        embeddings.append(sentvec)

    return embeddings


if __name__ == "__main__":
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']

    word_vectors = [
        'glove',
        'fasttext',
        'word2vec',
        'psl',
        'ppxxl',
        'pnmt'
    ]

    word_counts_choice = [
        NO_CNT,
        # SIF_CNT,
        # WIKI_FULL_CNT,
        # WIKI_GLOVE_CNT
    ]

    similarities = [
        'avg_cosine',
        'max_jaccard',
        'dynamax_jaccard',
        'dynamax_otsuka',
        'dynamax_dice'
    ]

    norm_choice = [
        False,
        # True
    ]

    results = []

    experiments = list(itertools.product(word_vectors,
                                         similarities,
                                         word_counts_choice,
                                         norm_choice))

    logging.info('Running {0} experiments. Good luck! :)\n\n\n'.format(len(experiments)))

    for idx, experiment in enumerate(experiments):
        word_vec_name = experiment[0]
        sim_name = experiment[1]
        word_counts = experiment[2]
        norm = experiment[3]

        logging.info('Word vectors: {0}'.format(word_vec_name))
        logging.info('Word Counts : {0}'.format(word_counts.name))
        logging.info('Similarity: {0}'.format(sim_name))
        logging.info('Normalize: {0}'.format(norm))
        logging.info('BEGIN\n\n\n')

        params_senteval = {
            'task_path': PATH_TO_DATA
        }
        params_experiment = {
            'word_vec_name': word_vec_name,
            'word_count_name': word_counts.name,
            'word_count_path': word_counts.path,
            'similarity_name': sim_name,
            'norm': norm
        }
        params_senteval.update(params_experiment)
        params_senteval['similarity'] = get_similarity_by_name(
            sim_name)

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        result = se.eval(transfer_tasks)
        result_dict = {
            'param': params_experiment,
            'eval': result
        }
        results.append(result_dict)
        logging.info('END. Experiment #{0} saved\n\n\n'.format(idx + 1))
