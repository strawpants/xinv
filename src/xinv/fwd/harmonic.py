## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl 

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase

# class Annual(FwdOpbase):
#     def __init__(self,n,annual_x='x', cache=False):
#         """Setup a forward operator which represents annual terms of a signal"""
#         super().__init__(obs_dim = annual_x, unknown_dim="annual", cache=cache)
#         self._n = n
        
        
#         ## add semiannual attribute here
    
#     def _jacobian_impl(self,dain):
#         """Creates the Jacobian of the forward operator (note:linear operator)"""
#         xcoords = dain.coords[self._obsdim]
#         order = 'C'
#         jacobian=xr.DataArray(np.zeros([len(xcoords),self._n], order =order), dims=[self._obsdim,self._unkdim], name = "annual_jacobian",coords={self._obsdim:xcoords,self._unkdim:np.arange(self._n)})

#         omega_annual = (2*np.pi)* (xcoords)
        
#         if self._n>0:            
#             jacobian.loc[:,0]= np.cos(omega_annual)
#         if self._n>1:
#             jacobian.loc[:,1]= np.sin(omega_annual)
        
#         return jacobian
    
    
class Harmonic(FwdOpbase):
    def __init__(self,frequency,obs_dim="time",unknown_dim="harmonic",x0=None,delta_x=1):
        """Setup a forward operator which represents harmonic cosine and sine combination"""
                
        
        super().__init__(obs_dim = obs_dim, unknown_dim=unknown_dim)
        self._frequency = frequency
        self._x0= x0
        self._n = 2
        self._delta_x=delta_x
        
    
        
    
    def _jacobian_impl(self,dain):
        """Creates the Jacobian of the forward operator (note:linear operator)"""
        xcoords = dain.coords[self._obsdim]
        order = 'C'
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._n], order =order), dims=[self._obsdim,self._unkdim], name = "harmonic_jacobian",coords={self._obsdim:xcoords,self._unkdim:np.arange(self._n)})

        if self._x0 is None:
            self._x0= xcoords.mean().item()
        
            omega_t = self._frequency*((xcoords - self._x0)/self._delta_x)
            
        
        if self._n>0:            
            jacobian.loc[:,0]= np.cos(omega_t)
        if self._n>1:
            jacobian.loc[:,1]= np.sin(omega_t)
        
        return jacobian
    
class Annual_harmonic(Harmonic):
    def __init__(self,x0,obs_dim="time",delta_x= np.timedelta64(365, 'D') + np.timedelta64(6, 'h')):
        
        super().__init__(frequency = 2*np.pi,unknown_dim="harmonic_annual",obs_dim=obs_dim,delta_x=delta_x)
        
class Semiannual_harmonic(Harmonic):
    def __init__(self,x0,obs_dim="time",delta_x= np.timedelta64(365, 'D') + np.timedelta64(6, 'h')):
        
        super().__init__(frequency = 4*np.pi,unknown_dim="harmonic_semiannual",obs_dim=obs_dim,delta_x=delta_x)