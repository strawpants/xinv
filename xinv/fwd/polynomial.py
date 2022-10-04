## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr
import numpy as np

class Polynomial:
    linear=True
    def __init__(self,n,outcoord=None):
        """Setup a forward operator which represents a polynomial of degree n"""
        self._n=n
        degs=[i for i in range(self._n+1)]
        self.incoord= xr.DataArray(degs,coords={"poly":degs},name='poly',dims='poly')
    
    def decorrelate(self,cov):
        pass

    def jacobian(self):
        pass

    def __call__(self,inpara,x):
        """implementation of the forward operator"""
        # todo: this could/should be optimized with cython
        out=[]
        degar=np.array([i for i in range(self._n+1)])
        for xelem in x:
            polycoef=np.power([xelem]*(self._n+1),degar)   
            out.append(polycoef@inpara)
        
        return xr.DataArray(out,coords={"x":x},name="poly_eval")


    # @property
    # def xcoord(self):
        # """return the coordinate of the x (unknown parameter) dimension"""
        # return xr.DataArray([i for i in range(self._n+1)],name='poly',dims='poly')
