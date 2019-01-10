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

import numpy as np
import sys
import logging

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

import utils
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from similarity.fuzzy import fbow_jaccard_factory

# Set up logger
logging.basicConfig(format='%(asctime)s : %(name)s : %(message)s', level=logging.DEBUG)


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
    results = []
    np.random.seed(1111)
    for word_vec_name in ['glove']:
        wv_path = utils.get_word_vec_path_by_name(word_vec_name)

        # U = utils.load_wordvec_matrix(wv_path, lo=0, hi=100000)
        # U = np.identity(300, dtype=np.float64)
        U = np.random.normal(size=(300, 300))

        fsimilarity = fbow_jaccard_factory(U)

        logging.info('Word vectors: {0}'.format(word_vec_name))
        logging.info('Similarity: {0}'.format('FBoW-Jaccard custom U'))
        logging.info('BEGIN\n\n\n')

        params_senteval = {
            'task_path': PATH_TO_DATA
        }
        params_experiment = {
            'word_vec_name': word_vec_name,
            'similarity_name': 'fbow_jaccard'
        }
        params_senteval.update(params_experiment)
        params_senteval['similarity'] = fsimilarity

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        result = se.eval(transfer_tasks)
        result_dict = {
            'param': params_experiment,
            'eval': result
        }
        results.append(result_dict)
