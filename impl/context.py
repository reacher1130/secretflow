import logging

import spu.libspu.link as link
from dotenv import load_dotenv
from util import GetParamEnv

import secretflow.distributed as sfd
from secretflow.distributed.primitive import DISTRIBUTION_MODE
from secretflow.ic.proxy import LinkProxy
from secretflow.utils.logging import LOG_FORMAT, get_logging_level, set_logging_level


class IcContext:

    def __init__(
        self, parties: list = None, mode: str = DISTRIBUTION_MODE.INTERCONNECTION
    ):
        self.algo = 'ecdh_psi'
        self.protocal_familes = 'ecc'
        self.version = '1'
        self.lctx = None  # yacl::link::Context


def MakeLink(start_transport, parties=None, self_rank=0):

    lctx = None
    try:
        lctx = link.CreateLinkContextForBlackBox(start_transport=start_transport)

    except Exception as e:
        lctx = link.CreateLinkContextForWhiteBox(parties, self_rank)

    return lctx


def CreateIcContext(start_transport=True) -> IcContext:
    ic_ctx = IcContext()
    ic_ctx.algo = GetParamEnv('algo')
    ic_ctx.protocal_familes = GetParamEnv('protocol_families')
    ic_ctx.lctx = MakeLink(start_transport=start_transport)
    return ic_ctx


def GetLabelRank(ic_ctx: IcContext):
    label_owner = GetParamEnv('label_owner')
    world_size = ic_ctx.lctx.world_size
    for i in range(world_size):
        if ic_ctx.lctx.party_by_rank(i) == label_owner:
            return i


def GetLabelName():
    return GetParamEnv('label_name')


class SgbHyperParam:
    def __init__(self):
        self.learn_rate = 0.001
        self.num_epoch = 1
        self.num_round = 5
        self.max_detph = 10
        self.bucket_eps = 0.08
        self.objective = 'logistic'
        self.reg_lambda = 0.001
        self.row_sample_by_tree = 0.9
        self.col_sample_by_tree = 0.9
        self.gamma = 1
        self.use_completely_sgb = False


def SuggestedSgbHyperParam():
    sgb_param = SgbHyperParam()
    sgb_param.use_completely_sgb = GetParamEnv('use_completely_sgb')
    sgb_param.num_epoch = GetParamEnv('num_epoch')
    sgb_param.num_round = GetParamEnv('num_round')
    sgb_param.max_detph = GetParamEnv('max_depth')
    sgb_param.bucket_eps = GetParamEnv('bucket_eps')
    sgb_param.objective = GetParamEnv('objective')
    sgb_param.reg_lambda = GetParamEnv('reg_lambda')
    sgb_param.row_sample_by_tree = GetParamEnv('row_sample_by_tree')
    sgb_param.col_sample_by_tree = GetParamEnv('col_sample_by_tree')
    sgb_param.gamma = GetParamEnv('gamma')
    sgb_param.learn_rate = GetParamEnv('learn_rate')

    return sgb_param


class SgbIoParam:
    def __init__(self):
        self.sample_size = 100000
        self.feature_nums = {}
        self.label_rank = -1
        self.label_name = ''


class HeProtocolParam:
    def __init__(self):
        self.sk_keeper = {}
        self.evaluators = {}
        self.he_parameters = {}


class SgbContext:

    def __init__(self, SgbHyperParam, SgbIoParam, HeProtocolParam):
        super().__init__()
        self.ic_ctx = None
        self.label_rank = 0
        self.label_name = ''
        self.sgb_param = SgbHyperParam
        self.sgb_io_param = SgbIoParam
        self.he_param = HeProtocolParam

    def HasLabel(self):
        return self.ic_ctx.l


def CreateSgbContext(ic_ctx: IcContext):
    sgb_ctx = SgbContext()
    sgb_ctx.sgb_param = SuggestedSgbHyperParam()
    sgb_ctx.sgb_io_param.label_rank = GetLabelRank(ic_ctx)
    sgb_ctx.sgb_io_param.label_name = GetLabelName()
    sgb_ctx.sgb_io_param.feature_nums = GetParamEnv('feature_nums')
    sgb_ctx.sgb_io_param.sample_size = GetParamEnv('sample_size')

    sgb_ctx.ic_ctx = ic_ctx
    return sgb_ctx
