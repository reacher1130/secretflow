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
        'self_party': 'bob',
    },
    'xgb': {
        'support_completely_sgb': True,
        'support_row_sample_by_tree': True,
        'support_col_sample_by_tree': True,
    },
    'heu': {
        "sk_keeper": {"party": "alice"},
        "evaluators": [{"party": "bob"}],
        # "mode": "PHEU",
        "he_parameters": {
            # ou is a fast encryption schema that is as secure as paillier.
            "schema": "ic-paillier",
        },
    },
}

ds = load_breast_cancer()
x, y = ds["data"], ds["target"]

dataset = {
    'features': {
        'alice': None,
        'bob': x[:, 15:],
    },
    'label': {
        'alice': None,
    },
}

from secretflow.ic.runner import run

run(config=config, dataset=dataset, logging_level='debug')
