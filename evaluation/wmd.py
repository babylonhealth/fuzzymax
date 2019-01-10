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
import logging
from gensim.models import KeyedVectors

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

import utils
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Set up logger
logging.basicConfig(format='%(asctime)s : %(name)s : %(message)s', level=logging.DEBUG)

#  Disable genism logging (too many WMD logs)
logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('smart_open').setLevel(logging.ERROR)


def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    return batch


if __name__ == "__main__":
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = []
    for word_vec_name in ['glove', 'word2vec', 'fasttext']:
        wv_path = utils.get_word_vec_path_by_name(word_vec_name)
        w2v_model = KeyedVectors.load_word2vec_format(wv_path, binary=False)

        logging.info('Word vectors: {0}'.format(word_vec_name))
        logging.info('Similarity: {0}'.format('wmd'))
        logging.info('BEGIN\n\n\n')

        params_senteval = {
            'task_path': PATH_TO_DATA
        }
        params_experiment = {
            'word_vec_name': word_vec_name,
            'similarity_name': 'wmd'
        }
        params_senteval.update(params_experiment)
        params_senteval['similarity'] = w2v_model.wmdistance

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        result = se.eval(transfer_tasks)
        result_dict = {
            'param': params_experiment,
            'eval': result
        }
        results.append(result_dict)
