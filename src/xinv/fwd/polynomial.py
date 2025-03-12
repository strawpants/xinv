## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase

class Polynomial(FwdOpbase):
    def __init__(self,n,poly_x='x',x0=None,delta_x=1,cache=False,unknown_dim="poly"):
        """Setup a forward operator which represents a polynomial of degree n"""
        super().__init__(obs_dim=poly_x,unknown_dim=unknown_dim,cache=cache)
        self._n=n
        self._unkcoord=np.arange(self._n+1)
        self._delta_x=delta_x
        self._x0=x0

    def _jacobian_impl(self,dain):
        """Creates the Jacobian of the forward operator (note:linear operator)"""
        xcoords=dain.coords[self._obsdim]
        if self._x0 is None:
            self._x0=xcoords.mean().item()
        
        #normalize xcoords to proposed time scale

        order='C'
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._n+1],order=order),dims=[self._obsdim,self._unkdim],name="poly_jacobian",coords={self._obsdim:xcoords,self._unkdim:self._unkcoord})
        
        xcoords_rel=((xcoords-self._x0)/self._delta_x).data
        #fill polynomial scales
        for i in range(self._n+1):
            jacobian.loc[:,i]=np.power(xcoords_rel,i)
        return jacobian



