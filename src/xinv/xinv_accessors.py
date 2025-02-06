## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr

from xinv.core.neq import solve as neqsolve
import numpy as np

@xr.register_dataarray_accessor("xi")
class InverseDaAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def build_normal(self,fwdop,ecov=1,apriori=None):
        """Builds a normal equation system from forward operators, data and an accompanying covariance"""
        
        #create the normal equation system
        dsneq=fwdop.build_normal(daobs=self._obj,ecov=ecov)
        return dsneq

@xr.register_dataset_accessor("xi")
class InverseDsAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def solve(self,inplace=False):
        return neqsolve(self._obj,inplace) #solve the normal equation system
