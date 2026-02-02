## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase

class Polynomial(FwdOpbase):
    def __init__(self,n,poly_x='x',x0=None,delta_x=1,cache=False,unknown_dim="poly",obs_dim=None,**kwargs):
        """Setup a forward operator which represents a polynomial of degree n"""
        if obs_dim is None:
            obs_dim=poly_x
        super().__init__(obs_dim=obs_dim,unknown_dim=unknown_dim,cache=cache,**kwargs)
        self._n=n
        self._unkcoord=np.arange(self._n+1)
        self._delta_x=delta_x
        self._x0=x0
        self._poly_x=poly_x

    def _jacobian_impl(self,**kwargs):
        """Creates the Jacobian of the forward operator (note:linear operator)"""
        #figure out the xcoords to use for the polynomial
        if self._poly_x in kwargs:
            xcoords=kwargs[self._poly_x]
            if type(xcoords) == list:
                xcoords=np.asarray(cxoords)
        elif "daobs" in kwargs:
            xcoords=kwargs['daobs'].coords[self._poly_x]
        else:
            raise ValueError(f"Polynomial Jacobian operator cannot figure out xcoord values, provide either dataarray 'daobs' or through a xcoords argument to the Jacobian")

        if self._x0 is None:
            self._x0=np.mean(xcoords).item()
       
        #possibly copy auxiliary coordinates from the input
        if "auxcoords" in kwargs:
            coords={ky:val for ky,val in kwargs['auxcoords'].items()}
        else:
            coords={}
        #make sure that the xcoords share the obsdim
        coords[self._poly_x]=(self._obsdim,xcoords.data)
        coords[self._unkdim]=(self._unkdim,self._unkcoord)
        

        order='C'
        jacobian=xr.DataArray(np.zeros([len(xcoords),self._n+1],order=order),dims=[self._obsdim,self._unkdim],name="poly_jacobian",coords=coords)
        
        #add some attributes
        # jacobian[self._obsdim].attrs["x0"]=self._x0
        # jacobian[self._unkdim].attrs["delta_x"]=self._delta_x

        #normalize xcoords to proposed time scale
        xcoords_rel=((xcoords-self._x0)/self._delta_x).data
        #fill polynomial scales
        for i in range(self._n+1):
            jacobian.loc[:,i]=np.power(xcoords_rel,i)
        return jacobian



