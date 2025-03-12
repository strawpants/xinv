## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase

class Polynomial(FwdOpbase):
    def __init__(self,n,poly_x='x',cache=False,unknown_dim="poly"):
        """Setup a forward operator which represents a polynomial of degree n"""
        super().__init__(obs_dim=poly_x,unknown_dim=unknown_dim,cache=cache)
        self._n=n
        self._unkcoord=np.arange(self._n+1)

    def _jacobian_impl(self,dain):
        """Creates the Jacobian of the forward operator (note:linear operator)"""
        xcoords=dain.coords[self._obsdim]
        order='C'
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._n+1],order=order),dims=[self._obsdim,self._unkdim],name="poly_jacobian",coords={self._obsdim:xcoords,self._unkdim:self._unkcoord})
        
        #fill polynomial scales
        for i in range(self._n+1):
            jacobian.loc[:,i]=np.power(xcoords,i)
        return jacobian



