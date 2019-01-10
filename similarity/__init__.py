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

from .baseline import *
from .classical import *
from .fuzzy import *
from .ablation import *
from .soft_card import *


NAME_TO_SIM = {
    # Baselines
    'avg_cosine': avg_cosine,
    'set_jaccard': set_jaccard,
    'bag_jaccard': bag_jaccard,
    'sc_jaccard': sc_jaccard,

    # Fuzzy set similarities
    'max_jaccard': max_jaccard,
    'dynamax_jaccard': dynamax_jaccard,
    'dynamax_otsuka': dynamax_otsuka,
    'dynamax_dice': dynamax_dice,

    # Ablation study similarities
    'avg_jaccard': avg_jaccard,
    'sum_jaccard': sum_jaccard,
    'max_cosine': max_cosine,
    'dynamax_cosine': dynamax_cosine
}


def get_similarity_by_name(sim_name):
    return NAME_TO_SIM[sim_name]
