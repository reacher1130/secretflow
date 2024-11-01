import argparse
import logging
import os

from context import CreateIcContext
from dotenv import load_dotenv
from impl_handler import SgbIcHandler
from link_proxy import LinkProxy
from sklearn.datasets import load_breast_cancer
from util import *

import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ic.runner import run

# load_dotenv()
# 1. 环境变量
# 2. 根据环境变量设置参与方角色
# 3. 读取环境变量，读入csv数据转为dict
# 3. 运行


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default='/root/develop/ant-sf/secretflow/impl/data/breast_cancer_a.csv',
        help='input file',
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='/root/develop/ant-sf/secretflow/impl/data/breast_cancer_a.csv',
        help='output file',
    )
    # parser.add_argument()
    parser.add_argument(
        '--env_file',
        type=str,
        default='/root/develop/ant-sf/secretflow/impl/env/sgb-env-alice.env',
        help='env_file',
    )

    args = parser.parse_args()

    input_filename = get_input_filename(defult_file=args.input_file)
    output_filename = get_output_filename(defult_file=args.output_file)
    print(f'input_filename: {input_filename}')
    print(f'output_filename: {output_filename}')
    env_file = args.env_file
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting 互联互通 SGB...")

    # 不用握手来协调参数
    # try:
    #     # 创建链接

    #     pass
    #     # 初始化
    #     pass
    #     # 启动运行算法
    #     pass
    #     # 结束连接
    # except Exception as e:
    #     logging.error(f"Error occurred: {e}")

    logging.info("互联互通 SGB finished.")
    ds = load_breast_cancer()
    x, y = ds["data"], ds["target"]

    dataset = {
        'features': {
            'host.0': x[:, :15],
            'bob': None,
        },
        'label': {
            'alice': y,
        },
    }
    load_dotenv(env_file)
    for key, value in os.environ.items():
        print(f"{key}: {value}")

    config = {
        "algo": GetParamEnv('algo'),
        "learning_rate": GetParamEnv('learning_rate'),
        "num_epoch": GetParamEnv('num_epoch'),
        "protocol_families": GetParamEnv('protocol_families'),
        "start_transport": GetParamEnv('start_transport'),
        "num_round": GetParamEnv('num_round'),
        "max_depth": GetParamEnv('max_depth'),
        "bucket_eps": GetParamEnv('bucket_eps'),
        "objective": GetParamEnv('objective'),
        "reg_lambda": GetParamEnv('reg_lambda'),
        "row_sample_by_tree": GetParamEnv('row_sample_by_tree'),
        "col_sample_by_tree": GetParamEnv('col_sample_by_tree'),
        "gamma": GetParamEnv('gamma'),
        "use_completely_sgb": GetParamEnv('use_completely_sgb'),
        "sk_keeper": GetParamEnv('sk_keeper'),
        "evaluators": GetParamEnv('evaluators'),
        "he_parameters": GetParamEnv('he_parameters'),
        "feature_select": GetParamEnv('feature_select'),
    }
    # ctx = CreateIcContext()
    print(config)
    # if ctx:
    #     print("CreateIcContext success")
    #     party = P
    #     # 创建context成功, 并互相连接
    #     # 启动每一方party
    # # run()

    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.INTERCONNECTION)
    LinkProxy.init()
    print("LinkProxy init success")
    config = {
        'xgb': {
            "num_round": GetParamEnv('num_round'),
            "max_depth": GetParamEnv('max_depth'),
            "bucket_eps": GetParamEnv('bucket_eps'),
            "objective": GetParamEnv('objective'),
            "reg_lambda": GetParamEnv('reg_lambda'),
            "row_sample_by_tree": GetParamEnv,
            "col_sample_by_tree": GetParamEnv('col_sample_by_tree'),
            "gamma": GetParamEnv('gamma'),
            "use_completely_sgb": GetParamEnv('use_completely_sgb'),
        },
        'heu': {
            "sk_keeper": GetParamEnv('sk_keeper'),
            "evaluators": GetParamEnv('evaluators'),
            "he_parameters": GetParamEnv('he_parameters'),
        },
    }
    handler = SgbIcHandler(config)
    handler.run_algo()

    LinkProxy.stop()
