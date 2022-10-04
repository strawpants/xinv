## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr
import numpy as np

class Polynomial:
    linear=True
    def __init__(self,n,x):
        """Setup a forward operator which represents a polynomial of degree n"""
        self._n=n
        self.ycoord="x"
        degs=[i for i in range(self._n+1)]
        

        self._ds= xr.Dataset(coords={"poly":degs,"x":x})


    def jacobian(self):
        """Creates the Jacobian of the forward operator (note:linear operator)"""
        if 'jacobian' not in self._ds:
            self._ds['jacobian']=(["x","poly"],np.zeros([self._ds.dims['x'],self._ds.dims['poly']]))
             
            xdim=self._ds.dims['x']
            pdim=self._ds.dims['poly']
            tmp=np.power(np.repeat(self._ds.x.values,pdim).reshape(xdim,pdim),self._ds.poly.values)
            self._ds['jacobian']=(["x","poly"],tmp)
        return self._ds.jacobian

    def __call__(self,inpara):
        """Apply the forward operator"""
        if type(inpara) is not xr.DataArray:
            inpara=xr.DataArray(inpara,dims=("poly"))
        out=self.jacobian()@inpara
        return xr.DataArray(out,coords={"x":self._ds.x},name="poly_eval")

    def unknown_coord(self):
        return self._ds.poly 
