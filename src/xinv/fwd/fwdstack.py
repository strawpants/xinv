## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase
from xinv.core.grouping import expand_as_group

class FwdStackOp(FwdOpbase):
    def __init__(self,fwdops=None,cache=False,unknown_dim="xinv_unk"):
        """Setup a forward operator which consists of stacking several other forward operators as groups which share the observation dimension"""
        super().__init__(cache=cache,unknown_dim=unknown_dim)
        self._fwdops=[]
        if hasattr(fwdops,'__iter__'):
            for fwdop in fwdops:
                self.append(fwdop)
        else:
            self.append(fwdops)

    def _jacobian_impl(self,dain):
        """Creates the Jacobian of the forward operator"""
        jacobian=None
        for stack_id,fwdop in enumerate(self._fwdops):
            #get Jacobian and add a multindex holding its stackid
            jacobian_i=fwdop.jacobian(dain)
            jacobian_i=expand_as_group(jacobian_i,group_dim=fwdop._unkdim,stack_dim=self._unkdim)
            if jacobian is None:
                jacobian=jacobian_i
            else:
                #Not really memory friendly at the moment, but ok for now
                jacobian=xr.concat([jacobian,jacobian_i],dim=self._unkdim)
        return jacobian

    def append(self,fwdop):
        if not isinstance(fwdop,FwdOpbase):
            raise ValueError("Input should be a FwdOpbase instance or derived class")
        if len(self._fwdops) == 0:
            #take the observational dimension from the first forward operator
            self._obsdim=fwdop._obsdim

        self._fwdops.append(fwdop)
        
