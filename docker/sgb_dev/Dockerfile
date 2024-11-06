FROM openanolis/anolisos:23 as builder

RUN yum install -y \
    wget gcc gcc-c++ autoconf bison flex git protobuf-devel libnl3-devel \
    libtool make pkg-config protobuf-compiler \
    && yum clean all

RUN cd / && git clone https://github.com/google/nsjail.git \
    && cd /nsjail && git checkout 3.3 -b v3.3 \
    && make && mv /nsjail/nsjail /bin

FROM secretflow/anolis8-python:3.10.13 as python

FROM openanolis/anolisos:23


COPY --from=builder /bin/nsjail /usr/local/bin/
COPY --from=python /root/miniconda3/envs/secretflow/bin/ /usr/local/bin/
COPY --from=python /root/miniconda3/envs/secretflow/lib/ /usr/local/lib/

RUN yum install -y protobuf libnl3 libgomp && yum clean all

RUN grep -rl '#!/root/miniconda3/envs/secretflow/bin' /usr/local/bin/ | xargs sed -i -e 's/#!\/root\/miniconda3\/envs\/secretflow/#!\/usr\/local/g'

COPY ./pkg/*.whl /tmp/

RUN pip install /tmp/*.whl && rm -rf /root/.cache

COPY .nsjail /root/.nsjail

# run as root for now
WORKDIR /root

# CMD ["/bin/bash", "-c", "python -m secretflow.impl.sgb_main"]
CMD ["python", "-m", "secretflow.impl.sgb_main"]

