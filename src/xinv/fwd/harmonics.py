## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl 

import numpy as np
import xarray as xr
from xinv.fwd import FwdOpbase

class Harmonics(FwdOpbase):
    def __init__(self,freqs,obs_dim="time",unknown_dim="harmonic",x0=None,delta_x=1):
        """ Setup a forward operator to represent multiple harmonic cosine and sine components."""
        
        super().__init__(obs_dim=obs_dim, unknown_dim=unknown_dim)
        self._freqs=freqs
        self._x0=x0
        self._delta_x=delta_x
        self._n=2*len(self._freqs)
        
    def _jacobian_impl(self,dain):
        """ Creates the Jacobian of the forward operator (note: linear operator)."""
        xcoords=dain.coords[self._obsdim]
        order='C'
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._n], order=order), dims=[self._obsdim,self._unkdim], name="harmonic_jacobian", coords={self._obsdim:xcoords,self._unkdim:np.arange(self._n)})
        
        if self._x0 is None:
            self._x0=xcoords.mean().item()          
            
        for i, freq in enumerate(self._freqs):
            omega_t=freq*(((xcoords-self._x0)/self._delta_x).astype(np.float64))
            jacobian.loc[:,2*i]=np.cos(omega_t)
            jacobian.loc[:,2*i+1]=np.sin(omega_t)
            
        return jacobian
    
class SeasonalHarmonics(Harmonics):
    def __init__(self,x0,semi_annual=True,obs_dim="time",delta_x=np.timedelta64(365,'D')+np.timedelta64(int(86400/4),'s')):
        
        freqs=[2*np.pi]
        if semi_annual is True:
            freqs.append(4*np.pi)
            
        super().__init__(freqs=freqs,unknown_dim="harmonics_seasonal",obs_dim=obs_dim,delta_x=delta_x)

        