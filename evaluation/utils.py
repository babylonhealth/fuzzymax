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

import io
import numpy as np
import logging
from sklearn.decomposition import TruncatedSVD


WORD_VEC_MAP = {
    'glove': 'glove.840B.300d.w2vformat.txt',
    'word2vec': 'GoogleNews-vectors-negative300.txt',
    'fasttext': 'fasttext-crawl-300d-2M.txt',
    'word2vec_skipgram': 'book_corpus_skip.txt',
    'word2vec_cbow': 'book_corpus_cbow.txt',
    'word2vec_conll': 'word2vec_conll17_skip.txt',
    'psl': 'paragram_300_sl999.w2vformat.txt',
    'ppxxl': 'paragram-phrase-XXL.w2vformat.txt',
    'pnmt': 'paragram-NMT.w2vformat.txt',
    'default': 'glove.840B.300d.w2vformat.txt'
}


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id,
                norm=False,
                path_to_counts=None):
    """
    Loads words and word vectors from a text file
    :param path_to_vec: path to word vector file in word2vec format
    :param word2id: words to load
    :param norm: normalise word vectors
    :param path_to_counts: path to word counts (enables SIF weights)
    :return: dict containing word: word vector
    """
    word_vec = {}
    word_freq_map = None
    if path_to_counts:
        word_freq_map = _get_word_freq_map(path_to_counts)

    with io.open(path_to_vec, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # always skip the first line, contains num of words and dim
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                np_vector = np.fromstring(vec, sep=' ')
                if norm:
                    np_vector = np_vector / np.linalg.norm(np_vector)
                if word_freq_map:
                    np_vector = _get_word_weight(word, word_freq_map) * np_vector
                word_vec[word] = np_vector

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


def _get_word_freq_map(path_to_counts):
    """
    Loads word counts and calculates word frequencies
    :param path_to_counts: path to word frequency file
    :return: dict containing word: word frequency
    """
    word_count_list = []

    total_count = 0.0
    with io.open(path_to_counts, 'r') as f:
        for line in f:
            word_count = line.split(' ')
            word = word_count[0]
            count = float(word_count[1])
            total_count += count
            word_count_list.append((word, count))

    word_freq_map = {}
    for word_count in word_count_list:
        word_freq_map[word_count[0]] = word_count[1] / total_count

    return word_freq_map


def _get_word_weight(word, word_freq_map, a=1e-3):
    """
    Computes SIF weight (Arora et al. 2017)
    :param word: input word
    :param word_freq_map: dict containing word: word freq.
    :param a: weight parameter
    :return: SIF weight for the word
    """
    word_freq = word_freq_map.get(word, 0.0)
    return a / (a + word_freq)


def get_word_vec_path_by_name(word_vec_name):
    base_path = '../data/word_vectors/'
    return base_path + WORD_VEC_MAP[word_vec_name]


def load_wordvec_matrix(path_to_vec, lo=0, hi=None):
    """
    Loads word vectors into a matrix
    :param path_to_vec: path to word vector file in word2vec format
    :param lo: start index
    :param hi: stop index
    :return: word vectors matrix
    """
    logging.info('Loading {0}'.format(path_to_vec))
    word_vec_list = []

    # if word2vec or fasttext file : skip first line "next(f)"
    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        next(f)
        for idx, line in enumerate(f):

            if idx < lo:
                continue
            if hi and idx >= hi:
                break
            word, vec = line.split(' ', 1)
            np_vector = np.fromstring(vec, sep=' ')
            word_vec_list.append(np_vector)
    logging.info('Loaded {0}, Vocab size: {1}'.format(path_to_vec,
                                                      len(word_vec_list)))
    return np.array(word_vec_list)


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc

    Function taken from https://github.com/PrincetonML/SIF
    Copyright (c) 2017 PrincetonML
    This function is licensed under the MIT license found at
    https://github.com/PrincetonML/SIF/blob/master/LICENSE.
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_
