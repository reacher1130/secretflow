from sklearn.datasets import load_breast_cancer

config = {
    'algo': 'sgb',
    'link': {
        'parties': {
            'alice': {
                # replace with alice's real address.
                'address': '127.0.0.1:9844',
                'listen_addr': '0.0.0.0:9844',
            },
            'bob': {
                # replace with bob's real address.
                'address': '127.0.0.1:9845',
                'listen_addr': '0.0.0.0:9845',
            },
        },
        'self_party': 'alice',
    },
    'xgb': {
        "num_round": 5,
        "max_depth": 5,
        "bucket_eps": 0.08,
        "objective": "logistic",
        "reg_lambda": 0.3,
        "row_sample_by_tree": 0.9,
        "col_sample_by_tree": 0.9,
        "gamma": 1,
        "use_completely_sgb": False,
    },
    'heu': {
        "sk_keeper": {"party": "alice"},
        "evaluators": [{"party": "bob"}],
        "he_parameters": {
            "schema": "ic-paillier",
            "key_pair": {
                "generate": {
                    # bit size should be 2048 to provide sufficient security.
                    "bit_size": 2048,
                },
            },
        },
    },
}

ds = load_breast_cancer()
x, y = ds["data"], ds["target"]

dataset = {
    'features': {
        'alice': x[:, :15],
        'bob': None,
    },
    'label': {
        'alice': y,
    },
}

import json
import os

from secretflow.ic.runner import run


def GetParamEnv(env_name: str) -> str:
    return os.getenv("runtime.compoment.parameter." + env_name)


def get_io_filename_from_env(input: bool) -> str:
    host_url = os.getenv("system.stortage")
    if not host_url or not host_url.startswith("file://"):
        host_url = os.getenv("system.stortage.host.url")
        if not host_url or not host_url.startswith("file://"):
            return None

    root_path = host_url[6:]
    json_str = (
        os.getenv("runtime.compoment.parameter.input.train_data")
        if input
        else os.getenv("runtime.compoment.parameter.output.train_data")
    )
    if not json_str:
        return None
    json_object = json.loads(json_str)
    relative_path = json_object["namespace"]
    file_name = json_object["name"]

    absolute_path = os.path.join(root_path, relative_path)
    if not input:
        os.makedirs(absolute_path, exist_ok=True)

    return os.path.join(absolute_path, file_name)


def get_input_filename_from_env() -> str:
    return get_io_filename_from_env(True)


def get_output_filename_from_env() -> str:
    return get_io_filename_from_env(False)


def get_input_filename(defult_file):
    input_filename = defult_file
    input_optional = get_input_filename_from_env()
    if input_optional is not None:
        input_filename = input_optional
    return input_filename


def get_output_filename(defult_file):
    output_filename = defult_file
    output_optional = get_output_filename_from_env()
    if output_optional is not None:
        output_filename = output_optional
    return output_filename


run(config=config, dataset=dataset, logging_level='debug')
