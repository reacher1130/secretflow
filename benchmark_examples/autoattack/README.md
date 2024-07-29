# 联邦攻防Benchmark

联邦攻防框架提供了自动调优工具（Secretflow Tuner），除了可以完成传统的automl能力，
还配合联邦学习的callback机制，实现攻防的自动调优。

用户实现攻击算法后，可以便捷地通过攻防框架，调整找到最合适的攻击参数、模型/数据拆分方法等，
可以借此来判断联邦算法的安全性。

在联邦算法的基础上，我们在几个经典的数据集+模型上，分别实现了几个攻击算法，并实现了benchmark，获得调优的结果。
目前支持的benchmark包括：

|    | datasets   | models   | auto_exploit | auto_fia | auto_lia | auto_norm | auto_replace | auto_replay |
|---:|:-----------|:---------|:-------------|:---------|:---------|:----------|:-------------|:------------|
|  0 | creditcard | dnn      | Support      | -        | -        | Support   | -            | -           |
|  1 | bank       | dnn      | Support      | Support  | Support  | Support   | Support      | Support     |
|  2 | bank       | deepfm   | -            | -        | -        | Support   | Support      | Support     |
|  3 | movielens  | dnn      | -            | -        | -        | -         | Support      | Support     |
|  4 | movielens  | deepfm   | -            | -        | -        | -         | Support      | Support     |
|  5 | criteo     | dnn      | Support      | -        | -        | Support   | Support      | Support     |
|  6 | criteo     | deepfm   | -            | -        | -        | Support   | Support      | Support     |
|  7 | mnist      | vgg16    | -            | Support  | Support  | -         | Support      | Support     |
|  8 | mnist      | resnet18 | -            | Support  | Support  | -         | Support      | Support     |
|  9 | drive      | dnn      | -            | Support  | -        | -         | -            | -           |
| 10 | cifar10    | vgg16    | -            | Support  | Support  | -         | Support      | Support     |
| 11 | cifar10    | resnet20 | -            | -        | Support  | -         | Support      | Support     |
| 12 | cifar10    | resnet18 | -            | Support  | Support  | -         | Support      | Support     |

## 如何添加新的实现

代码在`benchmark_example/autoattack`目录下。

`applications`目录下为具体的数据集+模型实现。
其中目录结构为`数据集分类/数据集名称/模型名称/具体实现`，例如`image/cifar10/vgg16`。
请继承自`ApplicationBase`并实现其所需的接口，并在最下层的`__init__.py`中，将主类重命名为`App`（参考已有实现）。

`attacks`目录下为具体的攻击实现。
请继承自`AttackCase`，并实现其中的抽象方法。
Attack中编写的是通用的攻击代码，如果攻击依赖于具体的数据集，例如需要辅助数据集和辅助模型，
则在application下的数据+模型代码中，提供这些代码。

## 运行单条测试

`benchmark_example/autoattack/main.py`提供了单个应用的测试入口，其必须参数为数据集、模型、攻击。
并可通过可选参数，控制是否使用GPU、是否简化测试等选项。
通过如下命令查看选项帮助：

```shell
cd secretflow
python benchmark_example/autoattack/main.py --help
```

以bank数据集，dnn数据集为例，可以通过`main.py`完成训练、攻击或自动调优攻击。
支持的数据集和攻击如上述的表格所示。

```shell
cd secretflow
# 训练
python benchmark_example/autoattack/main.py bank dnn train
# 攻击
python benchmark_example/autoattack/main.py bank dnn lia
# auto调优攻击
python benchmark_example/autoattack/main.py bank dnn auto_lia
```

## 运行benchmark

benchmark脚本提供了全部数据集下进行自动攻击调优的benchmark能力，并可以借助ray，实现gpu+分布式训练加速。

### 启动集群

benchmark脚本支持在单台机器上自动启动ray集群进行调优测试，
但通常其使用场景是在多台GPU机器下启动分布式ray集群，以加速实验。

首先需要参考Secretflow部署，在多台机器上完成conda环境安装，python环境安装，以及Secretflow的部署。
注意python版本必须一致。

我们首先要在机器上启动ray集群，并指定机器资源，其中包括：

- --gpu_nums选项：如果使用GPU，需要指定每台机器有几块GPU；
- 角色标签资源：如'alice'、'bob'，值和cpu数量相等即可；
- gpu内存资源：通过'gpu_mem'指定，单位为B，指定为该机器GPU总内存。


```shell
# 在首台机器上，启动ray头结点
ray start --head --port=6379 --resources='{"alice": 16, "bob":16, "gpu_mem": 85899345920}' --num-gpus=1 --disable-usage-stats --include-dashboard False
# 在其余机器上，启动ray并连接头结点
ray start --address="headip:6379" --resources='{"alice": 16, "bob":16, "gpu_mem": 85899345920}' --num-gpus=1 --disable-usage-stats
# 在头结点查看ray集群状态，看节点数量是否正确
ray status
```
### 启动benchmark

由于配置参数较多，通常可以通过指定配置文件(`config.yaml`)的方式来运行，配置文件格式和介绍如下：

```yaml
# application configurations.
applications:
  # which target to run (train/attack/auto).
  mode: auto
  # which dataset to run (all/bank/...)
  dataset: all
  # which model to run (all/bank/...)
  model: all
  # which attack to run (all/bank/...)
  attack: all
  # whether to run a simple test with small dataset and small epoch.
  simple: false
  # whether to use gpu to accelerate.
  use_gpu: false
  # whether using debug mode to run sf or not
  debug_mode: true
  # a random seed can be set to achieve reproducibility
  random_seed: ~
# path configurations.
paths:
  # the dataset store path, you can put the datasets here, or will auto download.
  datasets: ~
  # the autoattack result store path.
  autoattack_path: ~
# Resources configurations.
# Only needed when using sim mode and need to indicate the cpu/gpu nums manually.
resources:
  # how many CPUs do all your machines add up to.
  num_cpus: ~
  # how many CPUs do all your machines add up to (need applications.use_gpu = true).
  num_gpus: 2
# When there are multiple ray clusters in your machine, specify one to connect.
ray:
  # the existing ray cluster's address for connection (ip:port).
  address: ~
```

通过指定配置文件运行的方式如下，由于运行时间较长，建议使用nohup后台运行：

```shell
cd secretflow
# 如使用上述配置文件，会运行全部的case
nohup python benchmark_example/autoattack/benchmark.py --config="./benchmark_example/autoattack/config.yaml" > benchmark.log 2>&1 &
```

也可以通过命令行方式运行，具体的命令行请参考`--help`：

```shell
cd secretflow
# 帮助
python benchmark_example/autoattack/benchmark.py --help
# 指定dataset和gpu
python benchmark_example/autoattack/benchmark.py --dataset=all --use_gpu
```

如果命令行和配置文件同时指定，命令行添加的选项会覆盖配置文件的选项。
