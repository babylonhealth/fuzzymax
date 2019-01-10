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
# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval
from similarity import get_similarity_by_name

# Set up logger
logging.basicConfig(format='%(asctime)s : %(name)s : %(message)s', level=logging.DEBUG)


def prepare(params, samples):
    return


def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    return batch


similarities = [
        'set_jaccard',
        'bag_jaccard'
    ]

if __name__ == "__main__":
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']
    results = []
    for sim_name in similarities:
        logging.info('Similarity: {0}'.format(sim_name))
        logging.info('BEGIN\n\n\n')

        params_senteval = {
            'task_path': PATH_TO_DATA
        }
        params_experiment = {
            'similarity_name': sim_name
        }
        params_senteval.update(params_experiment)
        params_senteval['similarity'] = get_similarity_by_name(sim_name)

        se = senteval.engine.SE(params_senteval, batcher, prepare)
        result = se.eval(transfer_tasks)
        result_dict = {
            'param': params_experiment,
            'eval': result
        }
        results.append(result_dict)
