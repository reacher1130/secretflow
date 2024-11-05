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
import os
import sys
from datetime import datetime
from typing import Callable

# The dataset path of testing dataset, dataset which not exists will be downloaded.
_DATASETS_PATH = os.path.join(os.path.expanduser('~'), '.secretflow/datasets')
# The autoattack result path for logging.
_AUTOATTACK_PATH = os.path.join(os.path.expanduser('~'), '.secretflow/workspace')
# A bool value for whether we are running simple test cases.
# If set to True, a small number of epochs will be used and
#  the number of samples in the dataset will be reduced to make the testing faster.
_IS_SIMPLE_TEST = False
# Whether to use gpu or not.
_USE_GPU = False
# Whether enable debug mode.
_DEBUG_MODE = True
# The benchmark target mode, e.g. train/attack/auto
_BENCHMRAK_MODE = 'train'
# The benchmark target dataset, e.g. all/bank/criteo/...
_BENCHMARK_DATASET = 'all'
# The benchmark targe model, e.g. all/dnn/deepfm/...
_BENCHMARK_MODEL = 'all'
# The benchmark targe attack, e.g. all/norm/fia/...
_BENCHMARK_ATTACK = 'all'
<<<<<<< HEAD
# When using distributed computing, indicate the total amount of CPU resources
_NUM_CPUS = None
# When using distributed computing, indicate the total amount of GPU resources
_NUM_GPUS = None
=======
# The benchmark targe defense, e.g. all/grad_avg/...
_BENCHMARK_DEFENSE = 'all'
# All possible GPU configurations, e.g. {V100: 4000000000, A100: 1600000000}
_GPU_CONFIG = None
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
# When there is an existing ray cluster to connectiong, we can set the address in config file.
# Generally, this value does not require and ray will auto search,
# but if there are multiple ray clusters, a clear address is required.
_RAY_CLUSTER_ADDRESS = None
# To achieve reproducibility.
_RANDOM_SEED = None
_TUNER_START_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def _get_not_none(input: dict, key, default):
    """get value from input dict, if key not exist or key exist but value is none, return default value."""
    if key not in input or input[key] is None:
        return default
    else:
        return input[key]


def init_globalconfig(**kwargs):
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if 'config' in kwargs:
        _init_global_config_by_config_file(kwargs.pop('config'))
        if len(kwargs) > 0:
            logging.warning(
                "The parameters specified through the command line "
                "will override the parameters specified in the configuration file."
                f"The override params include {kwargs.keys()}"
            )
    if len(kwargs) > 0:
        _init_global_config_by_kwargs(**kwargs)
    print(
        "After init global configs, the final configs is:\n"
        f"_IS_SIMPLE_TEST: {_IS_SIMPLE_TEST}\n"
        f"_USE_GPU: {_USE_GPU}\n"
        f"_DEBUG_MODE: {_DEBUG_MODE}\n"
        f"_BENCHMRAK_MODE: {_BENCHMRAK_MODE}\n"
        f"_BENCHMARK_DATASET: {_BENCHMARK_DATASET}\n"
        f"_BENCHMARK_MODEL: {_BENCHMARK_MODEL}\n"
        f"_BENCHMARK_ATTACK: {_BENCHMARK_ATTACK}\n"
<<<<<<< HEAD
        f"_NUM_CPUS: {_NUM_CPUS}\n"
        f"_NUM_GPUS: {_NUM_GPUS}\n"
=======
        f"_BENCHMARK_DEFENSE: {_BENCHMARK_DEFENSE}\n"
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
        f"_DATASETS_PATH: {_DATASETS_PATH}\n"
        f"_AUTOATTACK_PATH: {_AUTOATTACK_PATH}\n"
        f"_RAY_CLUSTER_ADDRESS: {_RAY_CLUSTER_ADDRESS}\n"
        f"_RANDOM_SEED:{_RANDOM_SEED}"
    )


def _init_global_config_by_config_file(config_file: str):
    global _IS_SIMPLE_TEST
    global _AUTOATTACK_PATH
    global _USE_GPU
    global _DEBUG_MODE
    global _DATASETS_PATH
    global _BENCHMARK_DATASET
    global _BENCHMARK_MODEL
    global _BENCHMARK_ATTACK
<<<<<<< HEAD
    global _BENCHMRAK_MODE
    global _NUM_CPUS
    global _NUM_GPUS
    global _RAY_CLUSTER_ADDRESS
    global _RANDOM_SEED
    config_file = os.path.abspath(config_file)
    try:
        import yaml

        with open(config_file, 'r') as f:
            config: dict = yaml.load(f, Loader=yaml.Loader)
            applications = config.get('applications', None)
            if applications:
                _BENCHMRAK_MODE = _get_not_none(applications, 'mode', _BENCHMRAK_MODE)
                _BENCHMARK_DATASET = _get_not_none(
                    applications, 'dataset', _BENCHMARK_DATASET
                )
                _BENCHMARK_MODEL = _get_not_none(
                    applications, 'model', _BENCHMARK_MODEL
                )
                _BENCHMARK_ATTACK = _get_not_none(
                    applications, 'attack', _BENCHMARK_ATTACK
                )
                _IS_SIMPLE_TEST = _get_not_none(applications, 'simple', _IS_SIMPLE_TEST)
                _USE_GPU = _get_not_none(applications, 'use_gpu', _USE_GPU)
                _DEBUG_MODE = _get_not_none(applications, 'debug_mode', _DEBUG_MODE)
                _RANDOM_SEED = _get_not_none(applications, 'random_seed', _RANDOM_SEED)
            paths = config.get('paths', None)
            if paths:
                _DATASETS_PATH = _get_not_none(paths, 'datasets', _DATASETS_PATH)
                _AUTOATTACK_PATH = _get_not_none(
                    paths, 'autoattack_path', _AUTOATTACK_PATH
                )
            resources = config.get('resources', None)
            if resources:
                _NUM_CPUS = _get_not_none(resources, 'num_cpus', _NUM_CPUS)
                _NUM_GPUS = _get_not_none(resources, 'num_gpus', _NUM_GPUS)
            ray_config = config.get('ray', None)
            if ray_config:
                _RAY_CLUSTER_ADDRESS = _get_not_none(
                    ray_config, 'address', _RAY_CLUSTER_ADDRESS
                )
    except ImportError:
        raise ImportError(
            'PyYAML is required when set config files, try "pip insatll pyyaml" first.'
        ).with_traceback(sys.exc_info()[2])
=======
    global _BENCHMRAK_ENABLE_TUNE
    global _BENCHMARK_DEFENSE
    global _GPU_CONFIG
    global _RAY_CLUSTER_ADDRESS
    global _RANDOM_SEED
    config_file = os.path.abspath(config_file)
    config: dict = read_config(config_file)
    # applications
    applications = config.get('applications', {})
    _BENCHMRAK_ENABLE_TUNE = _get_not_none(
        applications, 'enable_tune', _BENCHMRAK_ENABLE_TUNE
    )
    _BENCHMARK_DATASET = _get_not_none(applications, 'dataset', _BENCHMARK_DATASET)
    _BENCHMARK_MODEL = _get_not_none(applications, 'model', _BENCHMARK_MODEL)
    _BENCHMARK_ATTACK = _get_not_none(applications, 'attack', _BENCHMARK_ATTACK)
    _BENCHMARK_DEFENSE = _get_not_none(applications, 'defense', _BENCHMARK_DEFENSE)
    _IS_SIMPLE_TEST = _get_not_none(applications, 'simple', _IS_SIMPLE_TEST)
    _USE_GPU = _get_not_none(applications, 'use_gpu', _USE_GPU)
    _DEBUG_MODE = _get_not_none(applications, 'debug_mode', _DEBUG_MODE)
    _RANDOM_SEED = _get_not_none(applications, 'random_seed', _RANDOM_SEED)
    # paths
    paths = config.get('paths', {})
    _DATASETS_PATH = _get_not_none(paths, 'datasets', _DATASETS_PATH)
    _AUTOATTACK_PATH = _get_not_none(paths, 'autoattack_path', _AUTOATTACK_PATH)
    # recources
    resources = config.get('resources', {})
    _GPU_CONFIG = _get_not_none(resources, 'gpu', _GPU_CONFIG)
    # ray config
    ray_config = config.get('ray', {})
    _RAY_CLUSTER_ADDRESS = _get_not_none(ray_config, 'address', _RAY_CLUSTER_ADDRESS)
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9


def _init_global_config_by_kwargs(**kwargs):
    global _IS_SIMPLE_TEST
    global _AUTOATTACK_PATH
    global _USE_GPU
    global _DEBUG_MODE
    global _DATASETS_PATH
    global _BENCHMARK_DATASET
    global _BENCHMARK_MODEL
    global _BENCHMARK_ATTACK
<<<<<<< HEAD
    global _BENCHMRAK_MODE
    global _NUM_CPUS
    global _NUM_GPUS
=======
    global _BENCHMRAK_ENABLE_TUNE
    global _BENCHMARK_DEFENSE
    global _GPU_CONFIG
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9
    global _RAY_CLUSTER_ADDRESS
    global _RANDOM_SEED
    _BENCHMRAK_MODE = kwargs.get('mode', _BENCHMRAK_MODE)
    _BENCHMARK_DATASET = kwargs.get('dataset', _BENCHMARK_DATASET)
    _BENCHMARK_MODEL = kwargs.get('model', _BENCHMARK_MODEL)
    _BENCHMARK_ATTACK = kwargs.get('attack', _BENCHMARK_ATTACK)
    _IS_SIMPLE_TEST = kwargs.get('simple', _IS_SIMPLE_TEST)
    _USE_GPU = kwargs.get('use_gpu', _USE_GPU)
    _DEBUG_MODE = kwargs.get('debug_mode', _DEBUG_MODE)
    _DATASETS_PATH = kwargs.get('datasets', _DATASETS_PATH)
    _AUTOATTACK_PATH = kwargs.get('autoattack_path', _AUTOATTACK_PATH)
    _GPU_CONFIG = kwargs.get('gpu_config', _GPU_CONFIG)
    _RAY_CLUSTER_ADDRESS = kwargs.get('ray_cluster_address', _RAY_CLUSTER_ADDRESS)
    _RANDOM_SEED = kwargs.get('random_seed', _RANDOM_SEED)


def get_dataset_path() -> str:
    global _DATASETS_PATH
    return _DATASETS_PATH


def get_autoattack_path() -> str:
    global _AUTOATTACK_PATH
    return _AUTOATTACK_PATH


def is_simple_test() -> bool:
    global _IS_SIMPLE_TEST
    if _IS_SIMPLE_TEST:
        logging.warning(
            'Running with simple test mode, your custom settings will not be applied!'
        )
    return _IS_SIMPLE_TEST


def is_use_gpu() -> bool:
    global _USE_GPU
    return _USE_GPU


<<<<<<< HEAD
def get_total_num_cpus():
    global _NUM_CPUS
    if _NUM_CPUS is None:
        return 32
    return _NUM_CPUS


def get_total_num_gpus():
    global _NUM_GPUS
    assert is_use_gpu(), f"When get gpu nums, must indicate use_gpu."
    if _NUM_GPUS is None:
        return 1
    return _NUM_GPUS


def get_benchmrak_mode():
    global _BENCHMRAK_MODE
    return _BENCHMRAK_MODE
=======
def is_enable_tune():
    global _BENCHMRAK_ENABLE_TUNE
    return _BENCHMRAK_ENABLE_TUNE
>>>>>>> 95547ade7047df593ec6bd1b61845f69527078a9


def get_benchmark_dataset():
    global _BENCHMARK_DATASET
    return _BENCHMARK_DATASET


def get_benchmark_model():
    global _BENCHMARK_MODEL
    return _BENCHMARK_MODEL


def get_benchmark_attack():
    global _BENCHMARK_ATTACK
    return _BENCHMARK_ATTACK


def is_debug_mode():
    global _DEBUG_MODE
    return _DEBUG_MODE


def get_gpu_config() -> dict | None:
    global _GPU_CONFIG
    return _GPU_CONFIG


def get_self_globals() -> dict:
    g = globals().copy()
    g = {
        k: v
        for k, v in g.items()
        if k.startswith("_") and not k.endswith('__') and not isinstance(v, Callable)
    }
    return g


def get_cur_experiment_result_path():
    global _TUNER_START_TIME
    global _AUTOATTACK_PATH
    return _AUTOATTACK_PATH + "/" + _TUNER_START_TIME


def get_ray_cluster_address():
    global _RAY_CLUSTER_ADDRESS
    return _RAY_CLUSTER_ADDRESS


def get_random_seed():
    global _RANDOM_SEED
    return _RANDOM_SEED
