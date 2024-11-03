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


from secretflow.ic.runner import run

run(config=config, dataset=dataset, logging_level='info')
