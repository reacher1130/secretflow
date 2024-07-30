# Copyright 2024 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kuscia.proto.api.v1alpha1.common_pb2 import DataColumn
from kuscia.proto.api.v1alpha1.datamesh.domaindata_pb2 import DomainData
from kuscia.proto.api.v1alpha1.kusciatask.kuscia_task_pb2 import (
    AllocatedPorts,
    ClusterDefine,
    Party,
    Port,
    Service,
)

from secretflow.kuscia.entry import convert_domain_data_to_individual_table
from secretflow.kuscia.ray_config import RayConfig
from secretflow.kuscia.sf_config import get_sf_cluster_config
from secretflow.kuscia.task_config import KusciaTaskConfig
from secretflow.spec.extend.cluster_pb2 import SFClusterDesc
from secretflow.spec.v1.data_pb2 import DistData, IndividualTable, TableSchema


def test_load_configs():
    kuscia_request_json = {
        "task_id": "secretflow-task-20240705104523-single-psi",
        "task_cluster_def": '{"parties":[{"name":"alice","role":"","services":[{"portName":"spu","endpoints":["secretflow-task-20240705104523-single-psi-0-spu.alice.svc"]},{"portName":"fed","endpoints":["secretflow-task-20240705104523-single-psi-0-fed.alice.svc"]},{"portName":"global","endpoints":["secretflow-task-20240705104523-single-psi-0-global.alice.svc:25815"]}]},{"name":"bob","role":"","services":[{"portName":"spu","endpoints":["secretflow-task-20240705104523-single-psi-0-spu.bob.svc"]},{"portName":"fed","endpoints":["secretflow-task-20240705104523-single-psi-0-fed.bob.svc"]},{"portName":"global","endpoints":["secretflow-task-20240705104523-single-psi-0-global.bob.svc:30818"]}]}],"selfPartyIdx":0,"selfEndpointIdx":0}',
        "allocated_ports": '{"ports":[{"name":"spu","port":25813,"scope":"Cluster","protocol":"GRPC"},{"name":"fed","port":25814,"scope":"Cluster","protocol":"GRPC"},{"name":"global","port":25815,"scope":"Domain","protocol":"GRPC"},{"name":"node-manager","port":25810,"scope":"Local","protocol":"GRPC"},{"name":"object-manager","port":25811,"scope":"Local","protocol":"GRPC"},{"name":"client-server","port":25812,"scope":"Local","protocol":"GRPC"}]}',
    }

    kuscia_config = KusciaTaskConfig.from_json(kuscia_request_json)

    assert kuscia_config.task_id == "secretflow-task-20240705104523-single-psi"
    assert len(kuscia_config.task_cluster_def.parties) == 2
    assert len(kuscia_config.task_allocated_ports.ports) == 6

    ray_config = RayConfig.from_kuscia_task_config(kuscia_config)
    assert (
        ray_config.ray_node_ip_address
        == "secretflow-task-20240705104523-single-psi-0-global.alice.svc"
    )
    assert ray_config.ray_gcs_port == 25815
    assert (
        ray_config.ray_min_worker_port
        >= 10000 & ray_config.ray_min_worker_port
        <= 20000
    )
    assert (
        ray_config.ray_max_worker_port
        >= 10000 & ray_config.ray_max_worker_port
        <= 20000
    )
    assert ray_config.ray_max_worker_port - ray_config.ray_min_worker_port == 100


def test_get_sf_cluster_config():
    sf_cluster_desc = SFClusterDesc(
        parties=["alice", "bob", "carol"],
        devices=[
            SFClusterDesc.DeviceDesc(name="spu", type="SPU", parties=["alice", "bob"])
        ],
    )

    kuscia_task_cluster_def = ClusterDefine(
        parties=[
            Party(
                name="alice",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.4"]),
                    Service(port_name="spu", endpoints=["1.2.3.4"]),
                    Service(port_name="global", endpoints=["0.0.0.0:1236"]),
                ],
            ),
            Party(
                name="bob",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.5"]),
                    Service(port_name="spu", endpoints=["1.2.3.5"]),
                ],
            ),
            Party(
                name="carol",
                services=[
                    Service(port_name="fed", endpoints=["1.2.3.6:2345"]),
                ],
            ),
        ],
        self_party_idx=0,
    )

    kuscia_task_allocated_ports = AllocatedPorts(
        ports=[
            Port(name="fed", port=1234),
            Port(name="spu", port=1235),
        ]
    )

    kuscia_config = KusciaTaskConfig(
        task_id="task_id",
        task_cluster_def=kuscia_task_cluster_def,
        task_allocated_ports=kuscia_task_allocated_ports,
        sf_cluster_desc=sf_cluster_desc,
    )

    sf_cluster_config = get_sf_cluster_config(kuscia_config)

    assert list(sf_cluster_config.public_config.ray_fed_config.addresses) == [
        "0.0.0.0:1234",
        "1.2.3.5:80",
        "1.2.3.6:2345",
    ]

    assert list(sf_cluster_config.public_config.spu_configs[0].addresses) == [
        "0.0.0.0:1235",
        "http://1.2.3.5:80",
    ]

    assert sf_cluster_config.private_config.self_party == "alice"
    assert sf_cluster_config.private_config.ray_head_addr == "0.0.0.0:1236"


def test_convert_domain_data_to_individual_table():
    domain_data = DomainData(
        name="input",
        author="alice",
        relative_uri="x/y/z",
        type="table",
        columns=[
            DataColumn(name="f1", type="int"),
            DataColumn(name="f2", type="double", comment="feature"),
            DataColumn(name="id", type="str", comment="id"),
            DataColumn(name="label", type="int", comment="label"),
        ],
    )

    meta = IndividualTable(
        schema=TableSchema(
            features=["f1", "f2", "id", "label"],
            feature_types=["int", "double", "str", "int"],
        ),
        line_count=-1,
    )

    expected_individual_table = DistData(
        name="input",
        type="sf.table.individual",
        data_refs=[DistData.DataRef(uri="x/y/z", party="alice", format="csv")],
    )
    expected_individual_table.meta.Pack(meta)

    assert (
        convert_domain_data_to_individual_table(domain_data)
        == expected_individual_table
    )
