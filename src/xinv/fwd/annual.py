## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl 

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase

class annual(FwdOpbase):
    def __init__(self,n,annual_x='x', cache=False):
        """Setup a forward operator which represents annual terms of a signal"""
        super().__init__(obs_dim = annual_x, unknown_dim="annual", cache=cache)
        self._n = n
        
    
    def _jacobian_impl(self,dain):
        """Creates the Jacobian of the forward operator (note:linear operator)"""
        xcoords = dain.coords[self._obsdim]
        order = 'C'
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._n], order =order), dims=[self._obsdim,self._unkdim], name = "annual_jacobian",coords={self._obsdim:xcoords,self._unkdim:np.arange(self._n)})

        omega_annual = (2*np.pi)/ (xcoords)
        
        if self._n>0:            
            jacobian.loc[:,0]= np.cos(omega_annual)
        if self._n>1:
            jacobian.loc[:,1]= np.sin(omega_annual)
        
        return jacobian