from context import CreateSgbContext, IcContext
from interconnection.handshake import entry_pb2


class Party:
    def __init__(self, ctx: IcContext):
        self._ctx = ctx

    def Run(self):
        self_rank = self._ctx.lctx.rank
        factory = self.CreateHandlerFactory()
        handler_v2 = factory.CreateHandler(self._ctx)
        recv_rank = 0
        if recv_rank == self_rank:
            handler_v2.PassiveRun()
        else:
            handler_v2.ActiveRun()


    def CreateHandlerFactory(self):

        if self._ctx.algo == entry_pb2.ALGO_TYPE_SGB:

            return SgbHandlerFactory()
        # run
