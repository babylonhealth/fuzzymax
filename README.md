> **Note**
> This repository is no longer actively maintained by Babylon Health. For further assistance, reach out to the paper authors.

# FuzzyMax

FuzzyMax is an evaluation framework and a collection of fuzzy set similarity measures for word vectors described in

Vitalii Zhelezniak, Aleksandar Savkov, April Shen, Francesco Moramarco, Jack Flann, and Nils Y. Hammerla, [*Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors, ICLR 2019.*](https://openreview.net/forum?id=SkxXg2C5FX)

## Similarity Measures

Word vectors alone are sufficient to achieve excellent performance on the semantic textual similarity tasks (STS) when sentence representations and similarity measures are derived using the ideas from fuzzy set theory.

The two important special cases described in the paper are **MaxPool-Jaccard**

```python
import numpy as np


def max_jaccard(x, y):
    """
    MaxPool-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    m_x = np.max(x, axis=0)
    m_x = np.maximum(m_x, 0, m_x)
    m_y = np.max(y, axis=0)
    m_y = np.maximum(m_y, 0, m_y)
    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union
```

and **DynaMax-Jaccard**

```python
import numpy as np


def fuzzify(s, u):
    """
    Sentence fuzzifier.
    Computes membership vector for the sentence S with respect to the
    universe U
    :param s: list of word embeddings for the sentence
    :param u: the universe matrix U with shape (K, d)
    :return: membership vectors for the sentence
    """
    f_s = np.dot(s, u.T)
    m_s = np.max(f_s, axis=0)
    m_s = np.maximum(m_s, 0, m_s)
    return m_s


def dynamax_jaccard(x, y):
    """
    DynaMax-Jaccard similarity measure between two sentences
    :param x: list of word embeddings for the first sentence
    :param y: list of word embeddings for the second sentence
    :return: similarity score between the two sentences
    """
    u = np.vstack((x, y))
    m_x = fuzzify(x, u)
    m_y = fuzzify(y, u)

    m_inter = np.sum(np.minimum(m_x, m_y))
    m_union = np.sum(np.maximum(m_x, m_y))
    return m_inter / m_union
```



## Dependencies

This code is written in Python 3. The requirements are listed in `requirements.txt`.
```
pip3 install -r requirements.txt
```


## Evaluation tasks

The experimental framework derived from [SentEval](https://github.com/facebookresearch/SentEval) evaluates the similarity measures on the following datasets:

| [STS 2012](https://www.cs.york.ac.uk/semeval-2012/task6/)   | [STS 2012](https://www.cs.york.ac.uk/semeval-2012/task6/) | [STS 2014](http://alt.qcri.org/semeval2014/task10/) | [STS 2015](http://alt.qcri.org/semeval2015/task2/) | [STS 2016](http://alt.qcri.org/semeval2016/task1/) |

To get all the datasets, run (in data/downstream/):
```bash
./get_sts_data.bash
```
This will automatically download and preprocess the downstream datasets, and store them in data/downstream (warning: for MacOS users, you may have to use p7zip instead of unzip).


## Experiments

Word vectors files must be in a word2vec txt format and are placed in `data/word_vectors/`.
The mapping from word vector model name to filename is found in `evaluation/utils.py`.
Word count files (if required) are placed in `data/misc/`.

```python

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
```

All the experiments are located in `evaluation`. They include

1. `classical.py` - classical Jaccard similarity for sets and multisets.
2. `conf_intervals.py` - evaluates DynaMax-Jaccard against avg.-cosine and computes 95% BCa confidence intervals for the delta in performance.
3. `fuzzy_eval` - DynaMax-Jaccard and Max-pool-Jaccard on all 6 word vectors. Can optionally enable SIF weights.
4. `sif.py` - SIF + PCA (Arora et al. 2017)
5. `wmd.py` - WMD (Kusner et al. 2015)


## Feedback and Contact:

If this code is useful to your research, please consider citing

```
@inproceedings{
zhelezniak2018dont,
title={Don't Settle for Average, Go for the Max: Fuzzy Sets and Max-Pooled Word Vectors},
author={Vitalii Zhelezniak and Aleksandar Savkov and April Shen and Francesco Moramarco and Jack Flann and Nils Y. Hammerla},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=SkxXg2C5FX},
}
```

Contact: Vitalii Zhelezniak <vitali.zhelezniak@babylonhealth.com>

## Related work
* [S. Arora, Y. Liang, T. Ma - A Simple but Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017](https://openreview.net/pdf?id=SyK00v5xx)
* [D. Cer, Y. Yang, S. Kong, N. Hua, N. Limtiaco, R. St. John, N. Constant, M. Guajardo-Cespedes, S. Yuan, C. Tar, Y. Sung, B. Strope, R. Kurzweil - Universal Sentence Encoder, 2018](https://arxiv.org/abs/1803.11175)
* [A. Conneau, D. Kiela, L. Barrault, H. Schwenk, A. Bordes - Supervised Learning of Universal Sentence Representations from Natural Language Inference Data, EMNLP 2017](https://arxiv.org/abs/1705.02364)
* [A. Conneau, D. Kiela, SentEval: An Evaluation Toolkit for Universal Sentence Representations](https://arxiv.org/abs/1803.05449)
* [S. Jimenez, F. Gonzalez, A. Gelbukh. Text comparison using soft cardinality.](https://link.springer.com/chapter/10.1007%2F978-3-642-16321-0_31)
* [J. R Kiros, Y. Zhu, R. Salakhutdinov, R. S. Zemel, A. Torralba, R. Urtasun, S. Fidler - SkipThought Vectors, NIPS 2015](https://arxiv.org/abs/1506.06726)
* [M. J. Kusner, Y. Sun, N. I. Kolkin, K. Q. Weinberger. From word embeddings to document distances, ICML 2015](http://proceedings.mlr.press/v37/kusnerb15.pdf)
* [S. Subramanian, A. Trischler, Y. Bengio, C. J Pal - Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning, ICLR 2018](https://arxiv.org/abs/1804.00079)
* [J. Wieting, M. Bansal, K. Gimpel, K. Livescu. Towards Universal Paraphrastic Sentence Embeddings, ICLR 2016](https://arxiv.org/abs/1511.08198)
* [J. Wieting, K. Gimpel. Pushing the limits of paraphrastic sentence embeddings with millions of machine translations, ACL 2018](http://aclweb.org/anthology/P18-1042)
* [R. Zhao, K. Mao. Fuzzy bag-of-words model for document representation, IEEE Transactionson Fuzzy Systems](https://ieeexplore.ieee.org/document/7891009)
