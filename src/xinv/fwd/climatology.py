## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2026 Roelof Rietbroek

import numpy as np
import xarray as xr
from xinv.fwd import FwdOpbase
from xinv.core.logging import xinvlogger as logger

class Climatology(FwdOpbase):
    def __init__(self,time_var="time",freq='M',obs_dim=None,**kwargs):
        """ Setup a forward operator to represent Climatology (mean over certain periods."""
        if obs_dim is None:
            obs_dim=time_var
        
        self._freq=freq
        if self._freq == 'M':
            unknown_dim="month"
            self._nclim=12
        else:
            logger.error(f"{self.__class__.__name__}: Can not recognize request frequency: {self._freq}")
            raise NotImplementedError
        
        super().__init__(obs_dim=obs_dim, unknown_dim=unknown_dim,**kwargs)
        self._tvar=time_var

    def _jacobian_impl(self,**kwargs):
        """ Creates the Jacobian of the forward operator (note: linear operator)."""

        #figure out the xcoords to use for the polynomial
        if self._tvar in kwargs:
            #xcoords is directly supplied
            xcoords=kwargs[self._tvar]
        elif "daobs" in kwargs:
            xcoords=kwargs['daobs'].coords[self._tvar]
        else:
            raise ValueError(f"Harmonic Jacobian operator cannot figure out xcoord values, provide either dataarray 'daobs' or the explicit coordinate {self._tvar}=.. as an argument")
        order='C'
        
        
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._nclim], order=order), dims=[self._obsdim,self._unkdim], name="harmonic_jacobian", coords={self._obsdim:xcoords,self._unkdim:np.arange(1,self._nclim+1)})
        
        int_months=xcoords.dt.month.data
        
        for iclim in jacobian[self._unkdim].data:
            
            jacobian.loc[int_months == iclim,iclim]=1.0
         
        return jacobian
    

        
