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

import gc
import logging
import os
<<<<<<< HEAD
from typing import List
=======
import types
from typing import Callable, Dict, List
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9

import click
import torch.cuda

from benchmark_examples.autoattack.applications.base import ApplicationBase
from benchmark_examples.autoattack.attacks.base import AttackCase
from secretflow.tune.tune_config import RunConfig, TuneConfig

try:
    import secretflow as sf

    v = sf.version
except ImportError as e:
    print(
        "Cannot find secretflow module, "
        "maybe try use: "
        "export PYTHONPATH='/path/to/secretflow'"
    )
    raise e

import ray

import benchmark_examples.autoattack.utils.dispatch as dispatch
import secretflow as sf
import secretflow.distributed as sfd
from benchmark_examples.autoattack import global_config
from secretflow import PYU, tune
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.utils.errors import NotSupportedError

_PARTIES = ['alice', 'bob']


def show_helps():
    print("****** Benchmark need at least 3 args:")
    print("****** [1]: the dataset name, like 'cifar10', 'criteo', 'bank', etc.")
    print("****** [2]: the model name, like 'dnn', 'deepfm', etc.")
    print(
        "****** [3]: the run mode, need 'train','predict','[attack]' or 'auto_[attack]'."
    )
    print("****** example:")
    print("****** python benchmark_examples/autoattack/main.py drive dnn train")
    print("****** python benchmark_examples/autoattack/main.py drive dnn auto-fia")


def init_ray():
    ray.init(
        address=global_config.get_ray_cluster_address(),
        log_to_driver=True,
    )


def init_sf():
    debug_mode = global_config.is_debug_mode()
    sfd.set_distribution_mode(
        mode=DISTRIBUTION_MODE.SIMULATION if not debug_mode else DISTRIBUTION_MODE.DEBUG
    )
    sf.shutdown()
    address = global_config.get_ray_cluster_address()
    address = 'local' if address is None else address
    sf.init(
        _PARTIES,
        address=address,
        log_to_driver=True,
        omp_num_threads=os.cpu_count(),
        debug_mode=debug_mode,
    )
    alice = sf.PYU("alice")
    bob = sf.PYU("bob")
    return alice, bob


class AutoAttackResult:
    def __init__(self, results, best_results, metric_names, metric_modes):
        self.results = results
        self.best_results: List = best_results
        self.metric_names: List = metric_names
        self.metric_modes: List = metric_modes


<<<<<<< HEAD
def do_train(dataset: str, model: str, alice: PYU, bob: PYU):
    App = dispatch.dispatch_application(dataset, model)
    app: ApplicationBase = App({}, alice, bob)
    app.prepare_data()
    app.train()


def do_attack(dataset: str, model: str, attack: str, alice: PYU, bob: PYU):
    App = dispatch.dispatch_application(dataset, model)
    Attack = dispatch.dispatch_attack(attack)
    if attack not in App({}, alice, bob).support_attacks():
        raise NotSupportedError(
            f"Attack {attack} not supported in app {App.__name__}! "
            f"If not correct, check the implement of 'support_attacks' in class {App.__name__}"
=======
def objective_trainning(
    config: Dict,
    *,
    app: ApplicationBase,
    attack: AttackBase | None = None,
    defense: DefenseBase | None = None,
    origin_global_configs: Dict | None = None,
) -> Dict[str, float]:
    """
    The target function for ml train, attack, defense.
    This function will be executed remote by tune when use auto mode.
    Returns:
        Dict[str, float]: A dict type metrics with app train history and attack metrics.
    """
    if origin_global_configs:
        sync_remote_globals(origin_global_configs)
    attack = DefaultAttackCase() if attack is None else attack
    defense = DefaultDefenseCase() if defense is None else defense
    with (
        app as app,
        attack as attack,
        defense as defense,
    ):
        # set the tune config into app, attack and defense
        app.set_config(config)
        attack.set_config(config)
        defense.set_config(config)
        # first add defense callbacks, then add attack callbacks.
        attack_callback = attack.build_attack_callback(app)
        defense_callback: Callback = defense.build_defense_callback(app, attack)
        callbacks = [defense_callback, attack_callback]
        callbacks = [v for v in callbacks if v is not None]
        callbacks = None if len(callbacks) == 0 else callbacks
        train_metrics = app.train(callbacks=callbacks)
        get_metrics_params = tuple()
        attack_metrics_params = attack.attack_metrics_params()
        if attack_metrics_params is not None:
            preds = app.predict(callbacks=callbacks)
            get_metrics_params = (sf.reveal(preds), *attack_metrics_params)
        attack_metrics = (
            attack_callback.get_attack_metrics(*get_metrics_params)
            if attack_callback
            else {}
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
        )
    attack_case: AttackCase = Attack(alice, bob, App)
    attack_case.attack({})


def do_autoattack(dataset: str, model: str, attack: str, alice: PYU, bob: PYU):
    App = dispatch.dispatch_application(dataset, model)
    config_app: ApplicationBase = App({}, alice, bob)
    Attack = dispatch.dispatch_attack(attack)
    if attack not in config_app.support_attacks():
        raise NotSupportedError(
            f"Attack {attack} not supported in app {App.__name__}! "
            f"If not correct, check the implement of 'support_attacks' in class {App.__name__}"
        )
    attack_case: AttackCase = Attack(alice, bob, App, global_config.get_self_globals())
    search_space = attack_case.search_space()
    metric_names = attack_case.metric_name()
    metric_modes = attack_case.metric_mode()
    metric_names = (
        [metric_names] if not isinstance(metric_names, list) else metric_names
    )
<<<<<<< HEAD
    metric_modes = (
        [metric_modes] if not isinstance(metric_modes, list) else metric_modes
    )
    assert len(metric_names) == len(metric_modes)
    cluster_resources = config_app.resources_consumes()
=======
    return search_space


def _get_cluster_resources(
    app: ApplicationBase, attack: AttackBase | None, defense: DefenseBase | None
) -> List[Dict[str, float]] | List[List[Dict[str, float]]]:
    debug_mode = global_config.is_debug_mode()
    use_gpu = global_config.is_use_gpu()
    if not debug_mode and use_gpu:
        raise NotImplemented(
            "Does not support using GPU for trainning without debug_mode."
        )
    cluster_resources_pack = app.resources_consumption()
    if defense:
        cluster_resources_pack = defense.update_resources_consumptions(
            cluster_resources_pack, app, attack
        )
    if attack:
        cluster_resources_pack = attack.update_resources_consumptions(
            cluster_resources_pack, app
        )
    if debug_mode:
        cluster_resources = cluster_resources_pack.get_debug_resources()
    else:
        cluster_resources = cluster_resources_pack.get_all_sim_resources()
    logging.info(f"The preprocessed cluster resource = {cluster_resources}")
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
    if not global_config.is_use_gpu():
        cluster_resources = [cr.without_gpu() for cr in cluster_resources]
    else:
        cluster_resources = [
            cr.handle_gpu_mem(global_config.get_gpu_config())
            for cr in cluster_resources
        ]
<<<<<<< HEAD
    tuner = tune.Tuner(
        attack_case.attack,
        tune_config=TuneConfig(max_concurrent_trials=300),
        run_config=RunConfig(
            storage_path=global_config.get_cur_experiment_result_path(),
            name=f"{dataset}_{model}_{attack}",
        ),
        cluster_resources=cluster_resources,
        param_space=search_space,
    )
    results = tuner.fit()
    log_content = ""
=======
    return cluster_resources


def _get_metrics(
    results: ResultGrid,
    app: ApplicationBase,
    attack: AttackBase | None,
    defense: DefenseBase | None,
) -> AutoAttackResult:
    metrics = {}
    # app metrics need a prefix app_
    metrics.update({f"app_{k}": v for k, v in app.tune_metrics().items()})
    if attack:
        metrics.update(attack.tune_metrics())
    if defense:
        metrics.update(defense.tune_metrics(metrics))
    print(f"metricccc = {metrics}")
    log_content = f"BEST RESULT for {app} {attack} {defense}: \n"
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
    best_results = []
    for metric_name, metric_mode in zip(metric_names, metric_modes):
        best_result = results.get_best_result(metric=metric_name, mode=metric_mode)
<<<<<<< HEAD
        log_content += f"RESULT: {dataset}_{model}_{attack} attack {metric_name}'s best config(mode={metric_mode}) = {best_result.config}, "
        f"best metrics = {best_result.metrics},\n"
=======
        log_content += (
            f"  best config (name: {metric_name}, mode: {metric_mode}) = {best_result.config}\n"
            f"  best metrics = {best_result.metrics},\n"
        )
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
        best_results.append(best_result)
    logging.warning(log_content)
    return AutoAttackResult(results, best_results, metric_names, metric_modes)


<<<<<<< HEAD
def run_case(dataset: str, model: str, attack: str):
=======
def case_valid_check(
    dataset: str,
    model: str,
    attack: str | None,
    defense: str | None,
):
    app_cls, attack_cls, defense_cls = _dispatch_cls(dataset, model, attack, defense)
    app_impl: ApplicationBase = app_cls(alice=None, bob=None)
    attack_impl: AttackBase | None = None
    if attack_cls:
        attack_impl = attack_cls(alice=None, bob=None)
        if not attack_impl.check_app_valid(app_impl):
            # if attack not in app_impl.support_attacks():
            raise NotSupportedError(
                f"Attack {attack} not supported in app {app_impl}! "
                f"If not correct, check the implement of 'check_app_valid' in class {attack_cls}"
            )
    if defense_cls:
        defense_impl = defense_cls(alice=None, bob=None)
        if attack_impl and not defense_impl.check_attack_valid(attack_impl):
            raise NotSupportedError(
                f"Defense {defense} not supported in attack {attack}! "
                f"If not correct, check the implement of 'check_attack_valid' in class {defense_impl}"
            )
        if not defense_impl.check_app_valid(app_impl):
            raise NotSupportedError(
                f"Defense {defense} not supported in application {app_impl}! "
                f"If not correct, check the implement of 'check_app_valid' in class {defense_impl}"
            )


def run_case(
    dataset: str,
    model: str,
    attack: str | None,
    defense: str | None,
    enable_tune: bool = False,
    objective: Callable = objective_trainning,
):
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
    """
    Run a singal case with dataset, model and attack.
    """
    alice, bob = init_sf()
<<<<<<< HEAD
    if 'auto' in attack:
        init_ray()
        try:
            attack = attack.lstrip('auto_')
            return do_autoattack(dataset, model, attack, alice, bob)
        finally:
            ray.shutdown()
    elif attack == 'train':
        return do_train(dataset, model, alice, bob)
    else:
        return do_attack(dataset, model, attack, alice, bob)
=======
    app_cls, attack_cls, defense_cls = _dispatch_cls(dataset, model, attack, defense)
    app_impl: ApplicationBase = app_cls(alice=alice, bob=bob)
    attack_impl: AttackBase | None = (
        attack_cls(alice=alice, bob=bob) if attack_cls else None
    )
    defense_impl: DefenseBase | None = (
        defense_cls(alice=alice, bob=bob) if defense_cls else None
    )
    objective_name = f"{dataset}_{model}_{attack}_{defense}"
    # give ray tune a readable objective name.
    objective = types.FunctionType(objective.__code__, globals(), name=objective_name)
    try:
        if not enable_tune:
            return objective(
                {},
                app=app_impl,
                attack=attack_impl,
                defense=defense_impl,
                origin_global_configs=None,
            )
        else:
            if global_config.is_debug_mode():
                init_ray()

            search_space = _construct_search_space(app_impl, attack_impl, defense_impl)
            cluster_resources = _get_cluster_resources(
                app_impl, attack_impl, defense_impl
            )
            objective = tune.with_parameters(
                objective,
                app=app_impl,
                attack=attack_impl,
                defense=defense_impl,
                origin_global_configs=global_config.get_self_globals(),
            )
            tuner = tune.Tuner(
                objective,
                tune_config=TuneConfig(max_concurrent_trials=1000),
                run_config=RunConfig(
                    storage_path=global_config.get_cur_experiment_result_path(),
                    name=f"{dataset}_{model}_{attack}_{defense}",
                ),
                cluster_resources=cluster_resources,
                param_space=search_space,
            )
            results = tuner.fit()
            return _get_metrics(results, app_impl, attack_impl, defense_impl)
    finally:
        if global_config.is_debug_mode():
            ray.shutdown()
        else:
            sf.shutdown()
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9


@click.command()
@click.argument("dataset_name", type=click.STRING, required=True)
@click.argument("model_name", type=click.STRING, required=True)
<<<<<<< HEAD
@click.argument("run_mode", type=click.STRING, required=True)
=======
@click.argument("attack_name", type=click.STRING, required=False, default=None)
@click.argument("defense_name", type=click.STRING, required=False, default=None)
@click.option(
    "--enable_tune",
    is_flag=True,
    default=None,
    required=False,
    help='Whether to run in auto mode.',
)
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
@click.option(
    "--simple",
    is_flag=True,
    default=None,
    help='Whether to use simple testing for easy debugging.',
)
@click.option(
    "--debug_mode",
    type=click.BOOL,
    required=False,
    help='Wheter to run secretflow on the debug mode.',
)
@click.option(
    "--datasets_path",
    type=click.STRING,
    required=False,
    default=None,
    help='Datasets load path, default to "~/.secretflow/datasets"',
)
@click.option(
    "--autoattack_storage_path",
    type=click.STRING,
    required=False,
    default=None,
    help='Autoattack results storage path, default to "~/.secretflow/datasets"',
)
@click.option(
    "--use_gpu",
    is_flag=True,
    required=False,
    default=None,
    help="Whether to use GPU, default to False",
)
@click.option(
    "--ray_cluster_address",
    type=click.STRING,
    required=False,
    default=None,
    help="The existing ray cluster address to connect.",
)
@click.option(
    "--random_seed",
    type=click.STRING,
    required=False,
    default=None,
    help="To achieve reproducible.",
)
def run(
<<<<<<< HEAD
    dataset_name,
    model_name,
    run_mode,
    simple,
    debug_mode,
    datasets_path,
    autoattack_storage_path,
    use_gpu,
    ray_cluster_address,
    random_seed,
):
    """Run single case with dataset, model and attack(or autoattack)."""
    try:
        global_config.init_globalconfig(
            datasets_path=datasets_path,
            autoattack_storage_path=autoattack_storage_path,
            simple=simple,
            use_gpu=use_gpu,
            debug_mode=debug_mode,
            ray_cluster_address=ray_cluster_address,
            random_seed=random_seed,
        )
        run_case(dataset_name, model_name, run_mode)
    except Exception as e:
        show_helps()
        raise e
=======
    dataset_name: str,
    model_name: str,
    attack_name: str | None,
    defense_name: str | None,
    enable_tune: bool,
    simple: bool,
    debug_mode: bool,
    datasets_path: str | None,
    autoattack_storage_path: str | None,
    use_gpu: bool,
    ray_cluster_address: str | None,
    random_seed: int | None,
    config: str | None,
):
    """
    Run single case with dataset, model, attack and defense.\n
    ****** The command need at least 2 args (dataset, model):\n
    ****** [1]: the dataset name, like 'cifar10', 'criteo', 'bank', etc.\n
    ****** [2]: the model name, like 'dnn', 'deepfm', etc.\n
    ****** [3]: the attack name like 'norm', 'lia', etc., can be empty.\n
    ****** [4]: the defense name like 'grad', etc., can be empty.\n
    ****** --auto --config: if active auto mode (need a config file to know search spaces.).\n
    ****** example:\n
    ****** python benchmark_examples/autoattack/main.py bank dnn\n
    ****** python benchmark_examples/autoattack/main.py bank dnn norm\n
    ****** python benchmark_examples/autoattack/main.py ban dnn norm grad\n
    ****** python benchmark_examples/autoattack/main.py ban dnn norm --enable_tune --config="path/to/config.yamml"\n
    """
    global_config.init_globalconfig(
        datasets_path=datasets_path,
        autoattack_storage_path=autoattack_storage_path,
        simple=simple,
        use_gpu=use_gpu,
        debug_mode=debug_mode,
        ray_cluster_address=ray_cluster_address,
        random_seed=random_seed,
        config=config,
    )
    run_case(dataset_name, model_name, attack_name, defense_name, enable_tune)
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9


if __name__ == '__main__':
    run()
