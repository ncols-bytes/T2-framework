# Base on: https://github.com/sunlab-osu/TURL/blob/release_ongoing/utils/util.py

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
# limitations under the Licens

import json
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict, Counter
import pickle
import copy
import os

RESERVED_ENT_VOCAB = {0:{'wiki_id':'[PAD]', 'wiki_title':'[PAD]', 'count': -1, 'mid': -1},
                        1:{'wiki_id':'[ENT_MASK]','wiki_title':'[ENT_MASK]', 'count': -1, 'mid': -1},
                        2:{'wiki_id':'[PG_ENT_MASK]','wiki_title':'[PG_ENT_MASK]', 'count': -1, 'mid': -1},
                        3:{'wiki_id':'[CORE_ENT_MASK]','wiki_title':'[CORE_ENT_MASK]', 'count': -1, 'mid': -1}
                        }
RESERVED_ENT_VOCAB_NUM = len(RESERVED_ENT_VOCAB)

def load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=1):
    entity_vocab = copy.deepcopy(RESERVED_ENT_VOCAB)
    bad_title = 0
    few_entity = 0
    with open(os.path.join(data_dir, 'entity_vocab.txt'), 'r', encoding="utf-8") as f:
        for line in f:
            _, entity_id, entity_title, entity_mid, count = line.strip().split('\t')
            if ignore_bad_title and entity_title == '':
                bad_title += 1
            elif int(count) < min_ent_count:
                few_entity += 1
            else:
                entity_vocab[len(entity_vocab)] = {
                    'wiki_id': int(entity_id),
                    'wiki_title': entity_title,
                    'mid': entity_mid,
                    'count': int(count)
                }
    return entity_vocab

def load_type_vocab(data_dir, model_type, types):
    type_vocab = {}

    if model_type == 1:
        file_name = "type_vocab.txt"
        file_path = os.path.join(data_dir, file_name)
    elif model_type == 2:
        file_name = "type_vocab_f2.txt"
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                for idx, type in enumerate(types):
                    f.write(str(idx) + '\t' + type + '\n')
                    idx += 1
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    with open(file_path, "r") as f:
        for line in f:
            index, t = line.strip().split('\t')
            type_vocab[t] = int(index)
    return type_vocab
