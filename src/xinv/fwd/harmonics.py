## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl 

import numpy as np
import xarray as xr
from xinv.fwd import FwdOpbase

class Harmonics(FwdOpbase):
    def __init__(self, n, semi_annual= False, annual_x='x', cache=False):
        
        """Setup a forward operator that represents annual and semi-annual terms of a signal"""
        
        if not semi_annual:
            super().__init__(obs_dim=annual_x, unknown_dim="annual", cache=cache)
            self._n=n
        
        #elif semi_annual==True:
        else:
            super().__init__(obs_dim=annual_x, unknown_dim="annual_semi", cache=cache)
            self._n=n  # Number of annual + semiannual harmonics
        
        self._semi_annual=semi_annual

    def _jacobian_impl(self, dain):
        """Creates the Jacobian of the forward operator (linear operator)"""
        xcoords = dain.coords[self._obsdim]
        order = 'C'
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._n], order =order), dims=[self._obsdim,self._unkdim], name = "annual_jacobian",coords={self._obsdim:xcoords,self._unkdim:np.arange(self._n)})
        

        omega_annual =(2*np.pi)*xcoords # angular frequency
        phase_shift=0
        
        index=0            
        jacobian.loc[:,index]=np.cos(omega_annual+phase_shift)
        
        #index+=1

        if self._semi_annual:
            omega_semiannual = (4*np.pi)*xcoords 
            #phase_semi=np.arctan2(B,A)
            phase_semi=0
            jacobian.loc[:,index]=np.cos(omega_semiannual+phase_semi)


        return jacobian
