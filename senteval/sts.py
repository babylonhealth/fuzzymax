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

'''
STS-{2012,2013,2014,2015,2016} (unsupervised)
'''

from __future__ import absolute_import, division, unicode_literals

import io
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr
import scikits.bootstrap as bstrap
from senteval.utils import cosine


class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self, params, prepare):
        if 'similarity' in params:
            self.similarity = lambda s1, s2: np.nan_to_num(
                params.similarity(np.nan_to_num(s1), np.nan_to_num(s2)))
        else:  # Default similarity is cosine
            self.similarity = lambda s1, s2: np.nan_to_num(
                cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

        if self.compute_conf_intervals(params):
            assert 'baseline_similarity' in params
            self.baseline_similarity = lambda s1, s2: np.nan_to_num(
                params.baseline_similarity(np.nan_to_num(s1), np.nan_to_num(s2)))

        return prepare(params, self.samples)

    def run(self, params, batcher):
        seed = params.seed
        np.random.seed(seed)
        results = {}
        for dataset in self.datasets:
            sys_scores = []
            sys_scores_base = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(len(enc2)):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)

                        if self.compute_conf_intervals(params):
                            sys_score_base = self.baseline_similarity(enc1[kk], enc2[kk])
                            sys_scores_base.append(sys_score_base)

            results[dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}

            if self.compute_conf_intervals(params):
                r_sys = pearsonr(gs_scores, sys_scores)[0]
                r_sys_base = pearsonr(gs_scores, sys_scores_base)[0]

                data = list(zip(gs_scores, sys_scores, sys_scores_base))

                def statistic(data):
                    gs = data[:, 0]
                    sys = data[:, 1]
                    sysb = data[:, 2]
                    r1 = pearsonr(gs, sys)[0]
                    r2 = pearsonr(gs, sysb)[0]
                    return r1 - r2

                conf_int = bstrap.ci(data, statfunction=statistic, method='bca')
                results[dataset]['conf_int'] = {
                    'delta': r_sys - r_sys_base,
                    'conf_int': list(conf_int),
                    'baseline': r_sys_base
                }

                logging.debug('%s : r  = %.4f, r base = %.4f, delta = %.4f, CI = [%.4f, %.4f]' %
                              (dataset, r_sys, r_sys_base, r_sys - r_sys_base,
                               conf_int[0], conf_int[1]))

            else:
                logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                              (dataset, results[dataset]['pearson'][0],
                               results[dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                             dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                             dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)

        results['all'] = {'pearson': {'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results

    @staticmethod
    def compute_conf_intervals(params):
        return 'conf_intervals' in params and params['conf_intervals']


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN']
        self.loadFile(taskpath)


class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)


class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)


class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)
