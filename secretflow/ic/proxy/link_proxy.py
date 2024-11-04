# Copyright 2023 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Dict

import spu.libspu.link as link

from secretflow.ic.proxy.serializer import deserialize, serialize


class LinkProxy:
    self_party = None
    all_parties = None
    world_size = 2
    self_rank = 0
    recv_rank = 0

    _link = None
    _parties_rank = None

    @classmethod
    def init(cls):

        cls._link = link.CreateLinkContextForBlackBox(start_stransport=True)
        cls.self_rank = cls._link.rank
        logging.info(f'self rank: {cls.self_rank}')
        cls.self_party = cls._link.party_by_rank(cls.self_rank)
        logging.info(f'self party: {cls.self_party}')
        cls.world_size = cls._link.world_size
        cls.all_parties = [cls._link.party_by_rank(i) for i in range(cls.world_size)]
        logging.info(f'all parties: {cls.all_parties}')
        cls._parties_rank = {
            cls._link.party_by_rank(idx): idx for idx in range(cls.world_size)
        }
        logging.info(f'parties rank: {cls._parties_rank}')

    @classmethod
    def send_raw(cls, dest_party: str, msg_bytes: bytes):
        rank = cls._parties_rank[dest_party]
        cls._link.send_async(rank, msg_bytes)

    @classmethod
    def recv_raw(cls, src_party: str) -> bytes:
        # logging.info(f'cls._parties_rank: {cls._parties_rank}')
        rank = cls._parties_rank[src_party]
        return cls._link.recv(rank)

    @classmethod
    def send(cls, dest_party: str, data: Any):
        msg_bytes = serialize(data)
        cls.send_raw(dest_party, msg_bytes)
        # logging.debug(f'send type {type(data)} to {dest_party}')

    @classmethod
    def recv(cls, src_party: str) -> Any:
        # logging.info(f'recv from {src_party}')
        msg_bytes = cls.recv_raw(src_party)
        data = deserialize(msg_bytes)
        # logging.debug(f'recv type {type(data)} from {src_party}')
        return data

    @classmethod
    def stop(cls):
        cls._link.stop_link()
