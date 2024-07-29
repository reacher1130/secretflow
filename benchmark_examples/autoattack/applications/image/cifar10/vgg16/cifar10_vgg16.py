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

from torch import nn, optim
from torchmetrics import AUROC, Accuracy, Precision

from benchmark_examples.autoattack.applications.image.cifar10.cifar10_base import (
    Cifar10ApplicationBase,
)
from secretflow.ml.nn.applications.sl_vgg_torch import VGGBase, VGGFuse
from secretflow.ml.nn.utils import TorchModel, metric_wrapper, optim_wrapper


class Cifar10VGG16(Cifar10ApplicationBase):
    def __init__(self, config, alice, bob):
        super().__init__(
            config,
            alice,
            bob,
            hidden_size=4608,
            dnn_fuse_units_size=[512 * 3 * 3 * 2, 4096, 4096],
        )

    def _create_base_model(self):
        return TorchModel(
            model_fn=VGGBase,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-4),
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
        )

    def dnn_fuse_units_size_range(self):
        return [
            [512 * 3 * 3 * 2, 4096],
            [512 * 3 * 3 * 2, 4096, 4096],
            [512 * 3 * 3 * 2, 4096, 4096, 4096],
            [512 * 3 * 3 * 2, 4096, 4096, 4096, 4096],
        ]

    def create_base_model_alice(self):
        return self._create_base_model()

    def create_base_model_bob(self):
        return self._create_base_model()

    def create_fuse_model(self):
        return TorchModel(
            model_fn=VGGFuse,
            loss_fn=nn.CrossEntropyLoss,
            optim_fn=optim_wrapper(optim.Adam, lr=1e-4),
            metrics=[
                metric_wrapper(
                    Accuracy, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(
                    Precision, task="multiclass", num_classes=10, average='micro'
                ),
                metric_wrapper(AUROC, task="multiclass", num_classes=10),
            ],
            dnn_units_size=self.dnn_fuse_units_size,
        )

    def support_attacks(self):
        return ['lia', 'fia', 'replay', 'replace']

    def lia_auxiliary_model(self, ema=False):
        from benchmark_examples.autoattack.attacks.lia import BottomModelPlus

        bottom_model = VGGBase()
        model = BottomModelPlus(bottom_model, size_bottom_out=self.hidden_size)

        if ema:
            for param in model.parameters():
                param.detach_()

        return model
