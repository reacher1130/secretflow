import argparse

from util import *

from secretflow.ic.runner import run

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
    args = parser.parse_args()

    input_filename = get_input_filename(defult_file=args.input_file)
