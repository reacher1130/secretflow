import argparse
import ast
import logging
import os

from context import CreateIcContext
from dotenv import load_dotenv
from impl_handler import SgbIcHandler
from sklearn.datasets import load_breast_cancer
from util import *

import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ic.proxy.link_proxy import LinkProxy
from secretflow.utils.logging import LOG_FORMAT, get_logging_level, set_logging_level

# load_dotenv()
# 1. 环境变量
# 2. 根据环境变量设置参与方角色
# 3. 读取环境变量，读入csv数据转为dict
# 3. 运行


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env_file',
        type=str,
        default='/root/develop/ant-sf/secretflow/impl/env/sgb-env-alice.env',
        help='env_file',
    )
    args = parser.parse_args()
    env_file = args.env_file

    set_logging_level(level='debug')
    logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)

    logging.info("-----------Starting 互联互通 SGB...-----------")
    load_dotenv(env_file)
    logging.info("-----------sfd分布式模式: 互联互通-----------")
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.INTERCONNECTION)
    LinkProxy.init()
    logging.info("-----------建立连接, 初始化成功-----------")
    config = {
        'xgb': {
            "num_round": int(GetParamEnv('num_round')),
            "max_depth": int(GetParamEnv('max_depth')),
            "bucket_eps": float(GetParamEnv('bucket_eps')),
            "objective": str(GetParamEnv('objective')),
            "reg_lambda": float(GetParamEnv('reg_lambda')),
            "row_sample_by_tree": float(GetParamEnv('row_sample_by_tree')),
            "col_sample_by_tree": float(GetParamEnv('col_sample_by_tree')),
            "gamma": float(GetParamEnv('gamma')),
            "use_completely_sgb": bool(GetParamEnv('use_completely_sgb')),
        },
        'heu': {
            "sk_keeper": ast.literal_eval(GetParamEnv('sk_keeper')),
            "evaluators": ast.literal_eval(GetParamEnv('evaluators')),
            "he_parameters": ast.literal_eval(GetParamEnv('he_parameters')),
        },
    }

    handler = SgbIcHandler(config)
    handler.run_algo()

    LinkProxy.stop()
