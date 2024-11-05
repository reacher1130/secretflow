# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

<<<<<<< HEAD
from benchmark_examples.autoattack.attacks.base import AttackCase
=======
from benchmark_examples.autoattack.applications.base import (
    ApplicationBase,
    ClassficationType,
)
from benchmark_examples.autoattack.attacks.base import AttackBase, AttackType
from benchmark_examples.autoattack.utils.resources import ResourcesPack
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
from secretflow import reveal
from secretflow.ml.nn.sl.attacks.norm_torch import NormAttack


class NormAttackCase(AttackCase):
    def _attack(self):
        self.app.prepare_data()
        label = reveal(self.app.get_train_label().partitions[self.app.device_y].data)
        norm_callback = NormAttack(self.app.device_f, label)
        history = self.app.train(norm_callback)
        logging.warning(
            f"RESULT: {type(self.app).__name__} norm attack metrics = {norm_callback.get_attack_metrics()}"
        )
        return history, norm_callback.get_attack_metrics()

    def attack_search_space(self):
        # norm attack does not have search space.
        return {}

<<<<<<< HEAD
    def metric_name(self):
        return 'auc'

    def metric_mode(self):
        return 'max'
=======
    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        label = reveal(app.get_plain_train_label())
        return NormAttack(app.device_f, label)

    def attack_type(self) -> AttackType:
        return AttackType.LABLE_INFERENSE

    def tune_metrics(self) -> Dict[str, str]:
        return {'auc': 'max'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        # TODO: support multiclass
        return app.classfication_type() in [ClassficationType.BINARY]

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        return cluster_resources_pack
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
