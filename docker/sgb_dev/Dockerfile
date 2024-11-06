FROM ubuntu:latest as builder

# 安装必要的构建工具和依赖
RUN apt-get update && apt-get install -y \
    wget g++ autoconf bison flex git libprotobuf-dev libnl-3-dev libnl-genl-3-dev \
    libtool make pkg-config protobuf-compiler 
# && rm -rf /var/lib/apt/lists/*

RUN cd / && git clone https://github.com/google/nsjail.git \
    && cd nsjail && git checkout 3.3 -b v3.3 \
    && make && mv /nsjail/nsjail /bin

# FROM python:3.10.13 as python
FROM secretflow/anolis8-python:3.10.13 as python

FROM ubuntu:latest

# 从构建器阶段复制nsjail二进制文件
COPY --from=builder /bin/nsjail /usr/local/bin/

# 从Python阶段复制Python环境和库
COPY --from=python /root/miniconda3/envs/secretflow/bin/ /usr/local/bin/
COPY --from=python /root/miniconda3/envs/secretflow/lib/ /usr/local/lib/

# 安装在Ubuntu上对应的依赖
RUN apt-get update && apt-get install -y protobuf-compiler libnl-3-200 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 修改脚本中的路径
RUN grep -rl '#!/root/miniconda3/envs/secretflow/bin' /usr/local/bin/ | xargs sed -i -e 's/#!\/root\/miniconda3\/envs\/secretflow/#!\/usr\/local/g'

# 复制wheel包并安装
COPY ./pkg/*.whl /tmp/
RUN pip install /tmp/*.whl && rm -rf /root/.cache/pip

# 复制.nsajail配置文件
COPY .nsjail /root/.nsjail

# 设置工作目录并指定默认命令
WORKDIR /root
CMD ["/bin/bash", "-c", "python -m secretflow.impl.sgb_main"]
