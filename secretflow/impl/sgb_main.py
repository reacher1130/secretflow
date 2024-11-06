import argparse
import ast
import logging
import os

import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ic.proxy.link_proxy import LinkProxy
from secretflow.impl.core.impl_handler import SgbIcHandler
from secretflow.impl.core.util import *
from secretflow.utils.logging import LOG_FORMAT, get_logging_level, set_logging_level


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_mode',
        type=str,
        default='production',
        choices=['production', 'debug'],
        help='run_mode',
    )
    parser.add_argument(
        '--env_file',
        type=str,
        default='./env/interconnection.env',
        help='env_file',
        required=False,
    )
    args = parser.parse_args()
    if args.run_mode == 'debug':
        if args.env_file:
            from dotenv import load_dotenv

            load_dotenv(args.env_file)
            print(f"Using environment file: {args.env_file}")
        else:
            print("No environment file provided for debug mode.")

    set_logging_level(level='debug')
    logging.basicConfig(level=get_logging_level(), format=LOG_FORMAT)

    if GetParamEnv('algo') != 'sgb' or GetParamEnv('protocol_families') != 'phe':
        raise ValueError(
            f"Invalid algorithm and protocol families combination: {GetParamEnv('algo')} and {GetParamEnv('protocol_families')}"
        )

    logging.info("-----------Starting 互联互通 SGB...-----------")
    logging.info("-----------分布式引擎计算模式: 互联互通-----------")
    sfd.set_distribution_mode(mode=DISTRIBUTION_MODE.INTERCONNECTION)
    LinkProxy.init(start_transport=str_to_bool(GetParamEnv('start_transport')))
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
            "use_completely_sgb": str_to_bool(GetParamEnv('use_completely_sgb')),
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


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"unexpected exception")
        logging.shutdown()
        os._exit(1)
