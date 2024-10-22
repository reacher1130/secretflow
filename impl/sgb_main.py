import argparse
import logging

from dotenv import load_dotenv
from util import *

from secretflow.ic.runner import run

load_dotenv()
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
        help='output file',z
    )
    args = parser.parse_args()

    input_filename = get_input_filename(defult_file=args.input_file)
    output_filename = get_output_filename(defult_file=args.output_file)
    print(f'input_filename: {input_filename}')
    print(f'output_filename: {output_filename}')

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting 互联互通 SGB...")
    # 不用握手来协调参数
    try:
        # 创建链接

        pass
        # 初始化
        pass
        # 启动运行算法
        pass
        # 结束连接
    except Exception as e:
        logging.error(f"Error occurred: {e}")

    logging.info("互联互通 SGB finished.")
