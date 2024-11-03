# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import ast
import logging
from typing import List, Tuple

import pandas as pd
import spu
from link_proxy import LinkProxy
from util import *

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ic.handler.algo import xgb
from secretflow.ic.handler.protocol_family import phe
from secretflow.ml.boost.sgb_v import Sgb, SgbModel

# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class SgbIcHandler:
    def __init__(self, config: dict):
        self._dataset = dict()
        self._xgb = xgb.XgbConfig(config['xgb'])
        self._phe = phe.PheConfig(config['heu'])

    def run_algo(self):

        print('+++++++++++++++ run sgb ++++++++++++++++++')
        params = self._process_params()
        self._read_dataset()
        x, y = self._process_dataset()
        model = self._train(params, x, y)
        self._evaluate(model, x, y)

    def _process_params(self) -> dict:
        self_rank = LinkProxy.self_rank
        active_rank = LinkProxy.recv_rank

        params = {
            "enable_packbits": True,
            "batch_encoding_enabled": False,
            "tree_growing_method": "level",
            "num_boost_round": self._xgb.num_round,
            "max_depth": self._xgb.max_depth,
            "sketch_eps": self._xgb.bucket_eps,
            "rowsample_by_tree": self._xgb.row_sample_by_tree,
            "colsample_by_tree": self._xgb.col_sample_by_tree,
            "first_tree_with_label_holder_feature": self._xgb.use_completely_sgb,
        }

        if self_rank == active_rank:
            params.update(
                {
                    "objective": self._xgb.objective,
                    "reg_lambda": self._xgb.reg_lambda,
                    "gamma": self._xgb.gamma,
                }
            )

        return params

    def _read_dataset(self):
        input_file = get_input_filename(defult_file='../data/test_data.csv')
        chunk_size = 1000
        chunks = []
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        label_owner = GetParamEnv('label_owner')
        label_name = GetParamEnv('label_name')

        self_party = LinkProxy.self_party
        feature_select = ast.literal_eval(GetParamEnv('feature_select'))
        self._dataset['label'] = dict()
        self._dataset['label'].update({label_owner: None})
        if label_owner == self_party:
            self._dataset['label'][self_party] = df[label_name].values

        print(feature_select[self_party])
        self._dataset['features'] = dict()
        print(LinkProxy.all_parties)
        for party in LinkProxy.all_parties:
            if party != self_party:
                self._dataset['features'].update({party: None})
            else:

                self._dataset['features'].update(
                    {party: df.loc[:, feature_select[party]].values}
                )

    def _process_dataset(self) -> Tuple[FedNdarray, FedNdarray]:

        print('+++++++++++++++ process dataset ++++++++++++++++++')
        self_rank = LinkProxy.self_rank
        active_rank = LinkProxy.recv_rank
        self_party = LinkProxy.self_party
        active_party = LinkProxy.all_parties[active_rank]

        v_data = FedNdarray({}, partition_way=PartitionWay.VERTICAL)

        for party, feature in self._dataset['features'].items():
            if party == self_party:
                assert feature is not None
            party_pyu = sf.PYU(party)
            v_data.partitions.update({party_pyu: party_pyu(lambda: feature)()})

        assert active_party in self._dataset['label']
        y = self._dataset['label'][active_party]
        if self_party == active_party:
            assert y is not None
        party_pyu = sf.PYU(active_party)
        label_data = FedNdarray(
            {
                party_pyu: party_pyu(lambda: y)(),
            },
            partition_way=PartitionWay.VERTICAL,
        )

        return v_data, label_data

    def _train(self, params: dict, x: FedNdarray, y: FedNdarray) -> SgbModel:
        print(self._phe.config)
        heu = sf.HEU(self._phe.config, spu.spu_pb2.FM128)
        sgb = Sgb(heu)
        return sgb.train(params, x, y)

    @staticmethod
    def _evaluate(model: SgbModel, x: FedNdarray, y: FedNdarray):
        print('+++++++++++++++ evaluate ++++++++++++++++++')
        yhat = model.predict(x)

        yhat = reveal(yhat)
        y = reveal(list(y.partitions.values())[0])

        from sklearn.metrics import roc_auc_score

        print(f"auc: {roc_auc_score(y, yhat)}")
