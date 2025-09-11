## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl

import numpy as np
import xarray as xr
from xinv.core.attrs import rhs_attrs,N_attrs,ltpl_attrs,sigma0_attrs,nobs_attrs,npara_attrs,find_xinv_coords, xunk_coords_attrs,aux_coords_attrs,x0_attrs

from xinv.neq.build import build_normal as neq_build_normal

class FwdOpbase:
    def __init__(self,obs_dim=None,unknown_dim=None,cache=False,jacobname="jacobian",**bindargs):
        self._obsdim=obs_dim
        self._unkdim=unknown_dim
        self._unkdim_t=unknown_dim+"_"
        self._daobs=None
        self._cache_jacobian=cache
        self._jacob=None
        self._jacobname=jacobname
        #to be forwarded to the jacobian implementation together with additional arguments
        self._bindargs=bindargs

    @property
    def unkdim(self):
        """Return the unknown dimension"""
        return self._unkdim

    @property
    def obs_dim(self):
        """Return the observation dimension"""
        return self._obsdim
    
    @property
    def jacobname(self):
        """Return the name of the Jacobian variable"""
        return self._jacobname

    def jacobian(self,**kwargs): 
        """Create the Jacobian of the forward operator"""
        
        #possibly quickly return the cached Jacobian
        if self._jacob is not None and self._cache_jacobian:
            return self._jacob

        # if daobs is None and self._jacob is None:
            # raise ValueError("Requesting the Jacobian without arguments requires caching abilities of the forward operator")
        # elif daobs is None:
            # return self._jacob

        jacob=self._jacobian_impl(**self._bindargs,**kwargs)
        if type(jacob) == xr.DataArray:
            jacob.name=self._jacobname
            jacob=jacob.to_dataset()
        #add xunk_coord attributes to possible coordinate
        if self._unkdim in jacob.coords:
            jacob[self._unkdim].attrs.update(xunk_coords_attrs(state='linked'))
        

        if self._cache_jacobian:
            self._jacob=jacob
        return jacob

    def build_normal(self,daobs,ecov=1):
        #convenience function
        #just forward the call to the implementation
        return neq_build_normal(self,daobs,ecov)

    
    def __call__(self,inpara,**kwargs):
        """Apply the forward operator"""
        return self.jacobian(**kwargs)[self._jacobname]@inpara
