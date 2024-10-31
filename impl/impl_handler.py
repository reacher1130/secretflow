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
import logging
from typing import List, Tuple

import spu

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal
from secretflow.ic.handler.algo import xgb
from secretflow.ic.handler.protocol_family import phe
from secretflow.ic.proxy import LinkProxy
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
    def __init__(self, config: dict, dataset: dict):
        self._dataset = dataset
        self._xgb = xgb.XgbConfig(config['xgb'])
        self._phe = phe.PheConfig(config['heu'])

    def run_algo(self):

        print('+++++++++++++++ run sgb ++++++++++++++++++')
        params = self._process_params()
        x, y = self._process_dataset()
        model = self._train(params, x, y)
        self._evaluate(model, x, y)

    def _process_params(self) -> dict:
        self_party = LinkProxy.self_party
        active_party = LinkProxy.all_parties[0]

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

        if self_party == active_party:
            params.update(
                {
                    "objective": self._xgb.objective,
                    "reg_lambda": self._xgb.reg_lambda,
                    "gamma": self._xgb.gamma,
                }
            )

        return params

    def _process_dataset(self) -> Tuple[FedNdarray, FedNdarray]:
        self_party = LinkProxy.self_party
        active_party = LinkProxy.all_parties[0]

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
