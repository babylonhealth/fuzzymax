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

import numpy as np
from collections import Counter


def sc_jaccard(x, y):
    xUy = munion(x, y)
    sc_x = soft_cardinality(x)
    sc_y = soft_cardinality(y)
    sc_xUy = soft_cardinality(xUy)
    sc_uIv = sc_x + sc_y - sc_xUy
    return sc_uIv / sc_xUy


def soft_cardinality(s):
    norms = np.linalg.norm(s, axis=1)
    cs = np.dot(s, s.T) / norms / norms.T
    cs = np.clip(cs, a_min=0, a_max=None)
    cs = np.power(cs, 1)
    return np.sum(1.0 / np.sum(cs, axis=0))


def union(x, y):
    return np.unique(np.concatenate((x, y)), axis=0)


def ssum(x, y):
    return np.concatenate((x, y))


def munion(x, y):
    xc = Counter(tuple(r) for r in x)
    yc = Counter(tuple(r) for r in y)
    xUy = xc | yc
    return np.array([np.array(a) for a in xUy.elements()])
