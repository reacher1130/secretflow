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
import os.path
import uuid

import torch
import torch.nn as nn
import torch.optim as optim

from benchmark_examples.autoattack import global_config
from benchmark_examples.autoattack.attacks.base import AttackCase
from benchmark_examples.autoattack.global_config import is_simple_test
<<<<<<< HEAD
from secretflow import tune
=======
from benchmark_examples.autoattack.utils.data_utils import get_np_data_from_dataset
from benchmark_examples.autoattack.utils.resources import ResourcesPack
from secretflow.ml.nn.callbacks.attack import AttackCallback
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
from secretflow.ml.nn.core.torch import TorchModel, optim_wrapper
from secretflow.ml.nn.sl.attacks.fia_torch import FeatureInferenceAttack


class Generator(nn.Module):
    def __init__(self, attack_dim, victim_dim):
        super().__init__()
        self.victim_dim = victim_dim
        self.reshape = len(attack_dim) > 1

        input_shape = 1
        for aa in attack_dim:
            input_shape *= aa

        output_shape = 1
        for vv in victim_dim:
            output_shape *= vv

        input_shape += output_shape

        self.net = nn.Sequential(
            nn.Linear(input_shape, 600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(600, 200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, output_shape),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        if self.reshape:  # pic input
            bs = x.size(0)
            x = x.reshape(bs, -1)
            r = self.net(x)
            r = r.reshape([bs] + self.victim_dim)
            return r
        else:
            return self.net(x)


class FiaAttackCase(AttackCase):
    def _attack(self):
        self.app.prepare_data()
        victim_model_save_path = f'./sl_model_victim + {uuid.uuid4().int}'
        generator_save_path = f'./generator + {uuid.uuid4().int}'
        try:
            victim_model_dict = self.app.fia_victim_model_dict(victim_model_save_path)
            optim_fn = optim_wrapper(optim.Adam, lr=self.config.get('optim_lr', 0.0001))
            generator_model = TorchModel(
                model_fn=Generator,
                loss_fn=None,
                optim_fn=optim_fn,
                metrics=None,
                attack_dim=self.app.fia_attack_input_shape(),
                victim_dim=self.app.fia_victim_input_shape(),
            )
            data_buil = self.app.fia_auxiliary_data_builder()
            logging.warning(
                f"in this fia trail, the gloabdsk use gpu = {global_config.is_use_gpu()}"
            )
            fia_callback = FeatureInferenceAttack(
                victim_model_path=victim_model_save_path,
                attack_party=self.app.device_y,
                victim_party=self.app.device_f,
                victim_model_dict=victim_model_dict,
                base_model_list=[self.alice, self.bob],
                generator_model_wrapper=generator_model,
                data_builder=data_buil,
                victim_fea_dim=self.app.fia_victim_input_shape(),
                attacker_fea_dim=self.app.fia_attack_input_shape(),
                enable_mean=self.config.get('enale_mean', False),
                enable_var=True,
                mean_lambda=1.2,
                var_lambda=0.25,
                attack_epochs=self.config.get(
                    'attack_epochs', 1 if is_simple_test() else 5
                ),
                victim_mean_feature=self.app.fia_victim_mean_attr(),
                save_attacker_path=generator_save_path,
                exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
            )
            history = self.app.train(fia_callback)
            logging.warning(
                f"RESULT: {type(self.app).__name__} fia attack metrics = {fia_callback.get_attack_metrics()}"
            )
            return history, fia_callback.get_attack_metrics()
        finally:
            if os.path.exists(victim_model_save_path):
                os.remove(victim_model_save_path)
            if os.path.exists(generator_save_path):
                os.remove(generator_save_path)

    def attack_search_space(self):
        return {
            # 'attack_epochs': tune.search.grid_search([2, 5]),  # < 120
            'optim_lr': tune.search.grid_search([1e-3, 1e-4]),
        }

    def metric_name(self):
        return ['mean_model_loss', 'mean_guess_loss']

<<<<<<< HEAD
    def metric_mode(self):
        return ['min', 'min']
=======
class FiaAttackCase(AttackBase):
    """
    Fia attack needs:
    - attacker input shape (without batch size (first dim))
    - victim input shape (without batch size (first dim))
    - app.fia_auxiliary_data_builder(): An aux dataloader with some samples.
    """

    def __init__(self, alice=None, bob=None):
        super().__init__(alice, bob)

    def __str__(self):
        return 'fia'

    def build_attack_callback(self, app: ApplicationBase) -> AttackCallback:
        optim_fn = optim_wrapper(optim.Adam, lr=self.config.get('optim_lr', 0.0001))
        generator_model = TorchModel(
            model_fn=Generator,
            loss_fn=None,
            optim_fn=optim_fn,
            metrics=None,
            attack_dim=app.get_device_y_input_shape()[1:],
            victim_dim=app.get_device_f_input_shape()[1:],
        )
        device_y_sample_dataset = app.get_device_y_train_dataset(
            frac=0.4, enable_label=1, list_return=True
        )
        device_f_sample_dataset = app.get_device_f_train_dataset(
            frac=0.4, enable_label=1, list_return=True
        )
        alice_sample_dataset = (
            device_y_sample_dataset
            if app.device_y == app.alice
            else device_f_sample_dataset
        )
        bob_sample_dataset = (
            device_y_sample_dataset
            if app.device_y == app.bob
            else device_f_sample_dataset
        )

        data_buil = data_builder(
            alice_sample_dataset, bob_sample_dataset, app.train_batch_size
        )
        victim_sample_data = get_np_data_from_dataset(device_f_sample_dataset)
        victim_sample_data = victim_sample_data.mean(axis=0)
        victim_mean_attr = victim_sample_data.reshape(victim_sample_data.shape[0], -1)
        return FeatureInferenceAttack(
            attack_party=app.device_y,
            victim_party=app.device_f,
            base_model_list=[self.alice, self.bob],
            generator_model_wrapper=generator_model,
            data_builder=data_buil,
            victim_fea_dim=app.get_device_f_input_shape()[1:],
            attacker_fea_dim=app.get_device_y_input_shape()[1:],
            enable_mean=self.config.get('enale_mean', False),
            enable_var=True,
            mean_lambda=1.2,
            var_lambda=0.25,
            attack_epochs=self.config.get(
                'attack_epochs', 1 if is_simple_test() else 5
            ),
            victim_mean_feature=victim_mean_attr,
            exec_device='cuda' if global_config.is_use_gpu() else 'cpu',
        )

    def attack_type(self) -> AttackType:
        return AttackType.FEATURE_INFERENCE

    def tune_metrics(self) -> Dict[str, str]:
        return {'mean_model_loss': 'min', 'mean_guess_loss': 'min'}

    def check_app_valid(self, app: ApplicationBase) -> bool:
        return app.base_input_mode() in [InputMode.SINGLE]

    def update_resources_consumptions(
        self, cluster_resources_pack: ResourcesPack, app: ApplicationBase
    ) -> ResourcesPack:
        func = lambda x: x * 1.2
        return cluster_resources_pack.apply_debug_resources(
            'gpu_mem', func
        ).apply_sim_resources(app.device_y.party, 'gpu_mem', func)
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
