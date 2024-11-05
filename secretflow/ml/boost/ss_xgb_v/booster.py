# Copyright 2024 Ant Group Co., Ltd.
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
import math
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np

import secretflow as sf
from secretflow.data import FedNdarray
from secretflow.data.vertical import VDataFrame
from secretflow.device import (
    PYU,
    SPU,
    PYUObject,
    SPUCompilerNumReturnsPolicy,
    SPUObject,
    wait,
)
from secretflow.ml.boost.core.callback import (
    CallBackCompatibleModel,
    CallbackContainer,
    Checkpointing,
    TrainingCallback,
)
from secretflow.ml.boost.core.data_preprocess import validate
from secretflow.ml.boost.ss_xgb_v.checkpoint import (
    SSXGBCheckpointData,
    build_ss_xgb_model,
    ss_xgb_model_to_checkpoint_data,
)
from secretflow.ml.boost.ss_xgb_v.model import XgbModel

from .core import node_split as split_fn
from .core.node_split import RegType
from .core.tree_worker import XgbTreeWorker as Worker


class Xgb(CallBackCompatibleModel):
    """
    This method provides both classification and regression tree boosting (also known as GBDT, GBM)
    for vertical split dataset setting by using secret sharing.

    SS-XGB is short for secret sharing XGB.
    more details: https://arxiv.org/pdf/2005.08479.pdf

    Args:
        spu: secret device running MPC protocols

    """

    def __init__(self, spu: Union[SPU, List[SPU]]) -> None:
        if not isinstance(spu, list):
            spu = [spu]
        self.spu = spu
        self._best_iteration = None
        self._best_score = None
        self.save_best = False

    def _update_pred(self, tree: Dict[PYU, PYUObject], weight: SPUObject) -> None:
        assert len(tree) == len(self.x)

        weight_selects = list()
        for worker in self.workers:
            device = worker.device
            assert device in tree
            s = worker.predict_weight_select(self.x[device].data, tree[device])
            weight_selects.append(s.to(self.spu[0]))

        current = self.spu[0](split_fn.predict_tree_weight)(weight_selects, weight)
        pred = self.spu[0](
            split_fn.update_train_pred,
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=self.fragment_count,
            static_argnames=("fragments"),
        )(self.pred, current, fragments=self.fragment_count)

        self.pred = pred if isinstance(pred, list) else [pred]

        for f_idx in range(self.fragment_count):
            spu_idx = f_idx % self.spus
            spu = self.spu[spu_idx]
            self.pred[f_idx] = self.pred[f_idx].to(spu)

    def _prepare(
        self,
        params: Dict,
        dataset: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
    ) -> None:
        start = time.time()
        assert len(self.spu) > 0, "need at least one spu device"
        self.spus = len(self.spu)
        x, x_shape, y, y_shape = validate(dataset, label)

        self.samples = y_shape[0]
        self.x = x.partitions

        self.trees = int(params.pop("num_boost_round", 10))
        assert (
            1 <= self.trees <= 1024
        ), f"num_boost_round should in [1, 1024], got {self.trees}"

        self.depth = int(params.pop("max_depth", 5))
        assert (
            self.depth > 0 and self.depth <= 16
        ), f"max_depth should in [1, 16], got {self.depth}"

        self.lr = float(params.pop("learning_rate", 0.3))
        assert (
            self.lr > 0 and self.lr <= 1
        ), f"learning_rate should in (0, 1], got {self.lr}"

        obj = params.pop("objective", "logistic")
        assert obj in [
            e.value for e in RegType
        ], f"objective should in {[e.value for e in RegType]}, got {obj}"
        self.obj = RegType(obj)

        self.reg_lambda = float(params.pop("reg_lambda", 0.1))
        assert (
            self.reg_lambda >= 0 and self.reg_lambda <= 10000
        ), f"reg_lambda should in [0, 10000], got {self.reg_lambda}"

        self.subsample = float(params.pop("subsample", 1))
        assert (
            self.subsample > 0 and self.subsample <= 1
        ), f"subsample should in (0, 1], got {self.subsample}"

        self.colsample = float(params.pop("colsample_by_tree", 1))
        assert (
            self.colsample > 0 and self.colsample <= 1
        ), f"colsample_by_tree should in (0, 1], got {self.colsample}"

        self.base = float(params.pop("base_score", 0))
        assert (
            self.base >= -10 and self.base <= 10
        ), f"base_score should in [-10, 10], got {self.base}"

        sketch = params.pop("sketch_eps", 0.1)
        assert sketch > 0 and sketch <= 1, f"sketch_eps should in (0, 1], got {sketch}"
        self.buckets = math.ceil(1.0 / sketch)
        self.seed = int(params.pop("seed", 42))

        assert len(params) == 0, f"Unknown params {list(params.keys())}"

        all_features = x_shape[1]
        # split buckets_map into [256, 512]MB fragments. (for FM64 ABY3 or FM128 SEMI2K)
        rows_limit = 256 * (1024**2) / all_features / self.buckets / 16
        fragment_count = math.ceil(self.samples / rows_limit)
        fragment_count = math.ceil(fragment_count / self.spus) * self.spus

        self.workers = [Worker(idx, device=pyu) for idx, pyu in enumerate(x.partitions)]
        self.fragment_count = fragment_count

        logging.info(f"fragment_count {fragment_count}")

        if fragment_count == 1:
            y = y.device(lambda y: y.reshape((1, y.shape[0])))(y)
            y = [y]
        else:
            y = y.device(
                lambda y: np.array_split(
                    y.reshape((1, y.shape[0])), fragment_count, axis=1
                ),
                num_returns=fragment_count,
            )(y)

        self.frag_samples = sf.reveal([f.device(lambda f: f.shape[1])(f) for f in y])
        self.y = [
            y[f_idx].to(self.spu[f_idx % self.spus]) for f_idx in range(fragment_count)
        ]
        wait(self.y)
        logging.info(f"prepare time {time.time() - start}s")

    def _global_setup(self) -> None:
        start = time.time()
        dones = []
        for idx, worker in enumerate(self.workers):
            done = worker.global_setup(
                self.x[worker.device].data,
                self.buckets,
                self.seed + idx,
            )
            dones.append(done)
        wait(dones)
        logging.info(f"global_setup time {time.time() - start}s")

        start = time.time()
        self.bucket_map = []
        f_start = 0
        for f_idx in range(self.fragment_count):
            spu_idx = f_idx % self.spus
            f_samples = self.frag_samples[f_idx]
            fragments_pre_party = []
            for worker in self.workers:
                b = worker.build_bucket_map(f_start, f_samples)
                fragments_pre_party.append(b)
            f_start += f_samples
            fragment = self.spu[spu_idx](lambda fs: jnp.concatenate(fs, axis=1))(
                [f.to(self.spu[spu_idx]) for f in fragments_pre_party]
            )
            self.bucket_map.append(fragment)
            if spu_idx == self.spus - 1:
                wait(self.bucket_map[-self.spus :])
                logging.info(
                    f"build & infeed bucket_map fragments [{f_idx - self.spus + 1}, {f_idx}]"
                )
        logging.info(f"build & infeed bucket_map time {time.time() - start}s")

        start = time.time()
        self.pred = []
        for f_idx in range(self.fragment_count):
            f_samples = self.frag_samples[f_idx]
            spu_idx = f_idx % self.spus
            pred = self.spu[spu_idx](
                split_fn.init_pred, static_argnames=("base", "samples")
            )(base=self.base, samples=f_samples)
            self.pred.append(pred)
            if spu_idx == self.spus - 1:
                wait(self.pred[-self.spus :])
        logging.info(f"init_pred time {time.time() - start}s")

    def _tree_setup(self) -> None:
        col_buckets_choices = []
        works_buckets_count = []
        for pyu_work in self.workers:
            choices, count = pyu_work.tree_setup(self.colsample)
            works_buckets_count.append(count)
            col_buckets_choices.append(choices)

        for idx, worker in enumerate(self.workers):
            col_buckets_choices[idx] = worker.update_buckets_count(
                [col.to(worker.device) for col in works_buckets_count],
                col_buckets_choices[idx],
            )

        assert len(self.y) == len(self.pred)

        if self.colsample < 1:
            self.col_choices = []
            for spu in self.spu:
                choices = [c.to(spu, spu_vis="public") for c in col_buckets_choices]
                spu_choices = spu(lambda c: jnp.concatenate(c, axis=None))(choices)
                self.col_choices.append(spu_choices)
        else:
            self.col_choices = [None] * self.spus

        self.sub_choices = list()
        self.ghs = list()
        for f_idx in range(len(self.y)):
            spu_idx = f_idx % self.spus
            spu = self.spu[spu_idx]
            y = self.y[f_idx]
            pred = self.pred[f_idx]

            samples = self.frag_samples[f_idx]
            choices = math.ceil(samples * self.subsample)
            assert choices > 0, f"subsample {self.subsample} is too small"

            if choices < samples:
                sub_choices = self.workers[0].device(
                    lambda s, c: np.sort(np.random.choice(s, c, replace=False))
                )(samples, choices)
                # same as colsample above, keep choices in public.
                sub_choices = sub_choices.to(spu, spu_vis="public")
            else:
                sub_choices = None

            gh = spu(
                split_fn.tree_setup,
                static_argnames=("objective"),
            )(
                pred,
                y,
                sub_choices,
                objective=self.obj,
            )

            self.sub_choices.append(sub_choices)
            self.ghs.append(gh)

            if spu_idx == self.spus - 1:
                wait(self.ghs[-self.spus :])

    def train(
        self,
        params: Dict,
        dtrain: Union[FedNdarray, VDataFrame],
        label: Union[FedNdarray, VDataFrame],
        dump_function: Optional[Callable] = None,
        checkpoint_data: Optional[SSXGBCheckpointData] = None,
    ) -> XgbModel:
        """train on dtrain and label.

        Args:
            params: Dict
                booster params, details are as follows
            dtrain: {FedNdarray, VDataFrame}
                vertical split dataset.
            label: {FedNdarray, VDataFrame}
                label column.
            dump_function: Callable
                dump function, called after every iteration, used to save ss xgb checkpoint data.
            checkpoint_data: SSXGBCheckpointData
                ss xgb checkpoint data, used to resume training from previous checkpoint.

        booster params details:

            num_boost_round : int, default=10
                Number of boosting iterations.
                range: [1, 1024]
            'max_depth': Maximum depth of a tree.
                default: 5
                range: [1, 16]
            'learning_rate': Step size shrinkage used in update to prevents overfitting.
                default: 0.3
                range: (0, 1]
            'objective': Specify the learning objective.
                default: 'logistic'
                range: ['linear', 'logistic']
            'reg_lambda': L2 regularization term on weights.
                default: 0.1
                range: [0, 10000]
            'subsample': Subsample ratio of the training instances.
                default: 1
                range: (0, 1]
            'colsample_by_tree': Subsample ratio of columns when constructing each tree.
                default: 1
                range: (0, 1]
            'sketch_eps': This roughly translates into O(1 / sketch_eps) number of bins.
                default: 0.1
                range: (0, 1]
            'base_score': The initial prediction score of all instances, global bias.
                default: 0
            'seed': Pseudorandom number generator seed.
                default: 42

        Return:
            XgbModel
        """
        self._prepare(params, dtrain, label)
        self._global_setup()

        checkpoint_model = None
        if checkpoint_data is not None:
            checkpoint_model = build_ss_xgb_model(checkpoint_data, self.spu[0])

        call_backs = CallbackContainer([Checkpointing(dump_function=dump_function)])
        self.model = (
            checkpoint_model
            if checkpoint_model is not None
            else XgbModel(self.spu[0], self.obj, self.base)
        )

        call_backs.before_training(self)
        while len(self.model.trees) < self.trees:
            tree_index = len(self.model.trees)
            call_backs.before_iteration(self, tree_index)
            start = time.time()

            self._tree_setup()

            logging.info(
                f"epoch {len(self.model.trees)} tree_setup time {time.time() - start}s"
            )

            start = time.time()

            tree, weight = self._train_tree()
            self.model.trees.append(tree)
            self.model.weights.append(weight)

            call_backs.after_iteration(self, tree_index, [])
            if len(self.model.trees) < self.trees:
                self._update_pred(tree, weight)
                wait(self.pred)
            else:
                wait(list(tree.values()) + [weight])

            logging.info(
                f"epoch {len(self.model.trees) - 1} time {time.time() - start}s"
            )
        call_backs.after_training(self)
        return self.model

    def _train_level(self, nodes_s: List[SPUObject]) -> SPUObject:
        assert len(nodes_s) == len(self.y)

        n_cache = list()
        level_GHs = list()
        start = time.time()
        for f_idx in range(len(nodes_s)):
            spu_idx = f_idx % self.spus
            spu = self.spu[spu_idx]
            s = nodes_s[f_idx]
            if self.cache:
                cache = self.cache[f_idx]
            else:
                cache = []
            sub_choices = self.sub_choices[f_idx]
            gh = self.ghs[f_idx]
            buckets_map = self.bucket_map[f_idx]

            GH, cache = spu(
                split_fn.compute_gradient_sums,
                num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
                user_specified_num_returns=2,
            )(s, cache, self.col_choices[spu_idx], sub_choices, gh, buckets_map)

            n_cache.append(cache)
            level_GHs.append(GH)
            if spu_idx == self.spus - 1:
                wait([*n_cache[-self.spus :], *level_GHs[-self.spus :]])
                logging.info(
                    f"fragment[{f_idx - self.spus + 1}, {f_idx}] gradient sum time {time.time() - start}s"
                )
                start = time.time()

        self.cache = n_cache

        # merge GH to spu 0
        level_GHs = [GH.to(self.spu[0]) for GH in level_GHs]
        wait(level_GHs)

        spu_split_buckets = self.spu[0](
            split_fn.find_best_split_bucket,
            static_argnames="reg_lambda",
        )(level_GHs, reg_lambda=self.reg_lambda)

        wait(spu_split_buckets)
        lchild_ss = []
        for worker in self.workers:
            # In the final tree model, which party hold the split feature for tree nodes is public information.
            # so, we can reveal 'split_buckets' to each pyu.
            lchild_s = worker.do_split(spu_split_buckets.to(worker.device))
            lchild_ss.append(lchild_s.to(self.spu[0]))

        childs_s = self.spu[0](
            split_fn.get_child_select,
            num_returns_policy=SPUCompilerNumReturnsPolicy.FROM_USER,
            user_specified_num_returns=self.fragment_count,
            static_argnames=("fragments"),
        )(nodes_s, lchild_ss, fragments=self.fragment_count)

        childs_s = childs_s if isinstance(childs_s, list) else [childs_s]

        for f_idx in range(self.fragment_count):
            spu_idx = f_idx % self.spus
            spu = self.spu[spu_idx]
            childs_s[f_idx] = childs_s[f_idx].to(spu)

        wait(childs_s)

        return childs_s

    def _init_root(self) -> List[SPUObject]:
        root_s = []
        for f_idx in range(self.fragment_count):
            spu_idx = f_idx % self.spus
            spu = self.spu[spu_idx]
            samples = self.frag_samples[f_idx]
            s = spu(split_fn.root_select, static_argnames=("samples"))(samples=samples)
            root_s.append(s)
        return root_s

    def _train_tree(self) -> Tuple[Dict[PYU, PYUObject], SPUObject]:
        nodes_s = self._init_root()
        self.cache = None
        for level in range(self.depth + 1):
            if level < self.depth:
                # split nodes
                start = time.time()
                nodes_s = self._train_level(nodes_s)
                logging.info(f"level {level} time {time.time() - start}s")
            else:
                # leaf nodes
                start = time.time()
                assert len(nodes_s) == len(self.ghs)
                sums = []
                for f_idx in range(len(self.ghs)):
                    spu_idx = f_idx % self.spus
                    spu = self.spu[spu_idx]
                    ss = nodes_s[f_idx]
                    gh = self.ghs[f_idx]
                    sub_choices = self.sub_choices[f_idx]
                    sums.append(
                        spu(split_fn.sum_leaf)(ss, gh, sub_choices).to(self.spu[0])
                    )
                    if spu_idx == self.spus - 1:
                        wait(sums[-self.spus :])

                weight = self.spu[0](split_fn.get_weight)(
                    sums, self.reg_lambda, self.lr
                )

        tree = {w.device: w.tree_finish() for w in self.workers}
        return tree, weight

    def eval_set(
        self,
        _evals,
        _feval,
    ) -> List:
        """Evaluate a set of data.

        Parameters
        ----------
        evals :
            List of items to be evaluated.
        feval :
            Custom evaluation function.

        Returns
        -------
        result: List
            Evaluation result List.
        """
        # TODO(zoupeicheng.zpc): implement this.
        return []

    def set_save_best(self, _save_best: bool):
        # TODO(zoupeicheng.zpc): really implement this.
        self.save_best = False

    def best_iteration(self):
        return self._best_iteration

    def best_score(self):
        return self._best_score

    def set_best_iteration_score(self, iteration, score):
        self._best_iteration = iteration
        self._best_score = score

    def get_model(self):
        return self.model


def build_checkpoint(
    booster: Xgb,
    _: TrainingCallback.EvalsLog,
    x: VDataFrame,
    label_name: str,
) -> SSXGBCheckpointData:
    """Build checkpoint from booster and evals log."""
    xgb_model = booster.get_model()
    # eval logs is ignored as there is no effective logs at now.
    return ss_xgb_model_to_checkpoint_data(xgb_model, x, label_name)
