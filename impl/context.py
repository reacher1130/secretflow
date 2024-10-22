import logging
from util import GetParamEnv
import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ic.proxy import LinkProxy
from secretflow.utils.logging import LOG_FORMAT, get_logging_level, set_logging_level



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
        self.ic_ctx.lctx = 









    def MakeLink(self, ):
    


