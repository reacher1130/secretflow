import logging

import spu.libspu.link as link
from dotenv import load_dotenv
from util import GetParamEnv

import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ic.proxy import LinkProxy
from secretflow.utils.logging import LOG_FORMAT, get_logging_level, set_logging_level

load_dotenv("/root/develop/ant-sf/secretflow/impl/env/sgb-env-alice.env")


class IcContext:
    def __init__(self, parties: list, mode: str = DISTRIBUTION_MODE.SINGLE):
        self.suggestedAlgo = 'ecdh_psi'
        self.suggestedProtocolfamilies = 'ecc'
        self.version = '1'
        self.lctx = None


class Context:
    def __init__(self, parties: list, mode: str = DISTRIBUTION_MODE.SINGLE):
        self.ic_ctx = IcContext()

    def CreateIcContext(self):
        self.ic_ctx.suggestedAlgo = GetParamEnv('algo')
        self.ic_ctx.suggestedProtocolfamilies = GetParamEnv('protocol_families')
        self.ic_ctx.lctx = self.MakeLink()
        self.start_transport = True

    def MakeLink(self, parties=None, self_rank=0):

        lctx = None
        try:
            lctx = link.CreateLinkContextForBlackBox()

        except Exception as e:
            lctx = link.CreateLinkContextForWhiteBox(parties, self_rank)

        return lctx


create_context = Context()

ctx = create_context.MakeLink()
