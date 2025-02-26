## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl 

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase

class constant(FwdOpbase):
    def __init__(self,n,constant_x='x', cache=False):
        """Setup a forward operator which represents a constant term of a signal"""
        super().__init__(obs_dim = constant_x, unknown_dim="constant", cache=cache)
        self._n = n
        
    
    def _jacobian_impl(self,dain):
        """Creates the Jacobian of the forward operator (note:linear operator)"""
        xcoords = dain.coords[self._obsdim]
        order = 'C'
        jacobian=xr.DataArray(np.ones([len(xcoords),self._n], order =order), dims=[self._obsdim,self._unkdim], name = "const_jacobian",coords={self._obsdim:xcoords,self._unkdim:np.arange(self._n)})

        return jacobian