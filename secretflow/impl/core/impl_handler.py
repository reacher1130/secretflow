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

import secretflow as sf
from secretflow.data import FedNdarray, PartitionWay
from secretflow.device.driver import reveal, wait
from secretflow.ic.handler.algo import xgb
from secretflow.ic.handler.protocol_family import phe
from secretflow.ic.proxy.link_proxy import LinkProxy
from secretflow.impl.core.util import *
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
        logging.info("-----------运行SGB算法-----------")
        params = self._process_params()
        self._read_dataset()
        x, y = self._process_dataset()
        model = self._train(params, x, y)
        self._evaluate(model, x, y)

    def _process_params(self) -> dict:
        self_party = LinkProxy.self_party
        label_owner = GetParamEnv('label_owner')
        if label_owner is None:
            raise ValueError("Label owner must be specified.")
        if label_owner not in LinkProxy.all_parties:
            raise ValueError("Invalid label owner.")

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

        if self_party == label_owner:
            params.update(
                {
                    "objective": self._xgb.objective,
                    "reg_lambda": self._xgb.reg_lambda,
                    "gamma": self._xgb.gamma,
                }
            )

        return params

    def _read_dataset(self):
        input_file = get_input_filename(default_file='../data/test_data.csv')
        chunk_size = 1000
        chunks = []
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        logging.info("The size of the data file read is : {}".format(df.shape))
        label_owner = GetParamEnv('label_owner')
        label_name = GetParamEnv('label_name')
        feature_select = ast.literal_eval(GetParamEnv('feature_select'))

        self_party = LinkProxy.self_party
        self._dataset['label'] = {
            label_owner: df[label_name].values if label_owner == self_party else None
        }

        self._dataset['features'] = {
            party: (
                df.loc[:, feature_select[party]].values if party == self_party else None
            )
            for party in LinkProxy.all_parties
        }

    def _process_dataset(self) -> Tuple[FedNdarray, FedNdarray]:

        logging.info("+++++++++++++++ process dataset ++++++++++++++++++")
        self_party = LinkProxy.self_party
        label_owner = GetParamEnv('label_owner')
        assert label_owner in LinkProxy.all_parties

        v_data = FedNdarray({}, partition_way=PartitionWay.VERTICAL)

        for party, feature in self._dataset['features'].items():
            if party == self_party:
                assert feature is not None
            party_pyu = sf.PYU(party)
            v_data.partitions.update({party_pyu: party_pyu(lambda: feature)()})

        assert label_owner in self._dataset['label']
        y = self._dataset['label'][label_owner]
        if self_party == label_owner:
            assert y is not None
        party_pyu = sf.PYU(label_owner)
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
        model = sgb.train(params, x, y)
        self._save_model(model, x.partitions.keys())
        return model

    def _save_model(self, model: SgbModel, parties):
        logging.info("+++++++++++++++ save model ++++++++++++++++++")
        out_put_path = get_output_filename(default_file='./out_puts/')
        save_path_dict = {
            device: os.path.join(out_put_path, device.party) for device in parties
        }
        r = model.save_model(save_path_dict)
        wait(r)

    @staticmethod
    def _evaluate(model: SgbModel, x: FedNdarray, y: FedNdarray):
        logging.info("+++++++++++++++ evaluate ++++++++++++++++++")
        yhat = model.predict(x)

        yhat = reveal(yhat)
        y = reveal(list(y.partitions.values())[0])

        from sklearn.metrics import roc_auc_score

        logging.info(f"auc: {roc_auc_score(y, yhat)}")
