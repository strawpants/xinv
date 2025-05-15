## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr

from xinv.neq import solve as neqsolve
from xinv.neq import transform as neqtransform
from xinv.neq import reduce as neqreduce
from xinv.neq import fix as neqfix
from xinv.neq import set_apriori as neqset_apriori
from xinv.neq import add as neqadd
from xinv.neq import zeros as neqzeros
from xinv.core.grouping import get_group,reindex_groups,rename_groups

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
    
    def transform(self,fwdop):
        return neqtransform(self._obj,fwdop) 
    
    def solve(self,inplace=False):
        return neqsolve(self._obj,inplace) #solve the normal equation system
    
    def reduce(self,idx):
        return neqreduce(self._obj,idx) #reduce  parameters from the normal equation system
    
    def fix(self,idx):
        return neqfix(self._obj,idx) #remove parameters from the normal equation system
    
    def set_apriori(self,dapri):
        return neqset_apriori(self._obj,dapri) #change apriori values

    def add(self,dsneqother):
        return neqadd(self._obj,dsneqother) #add/merge another normal equation system

    def get_group(self,group_name):
        return get_group(self._obj,group_name)

    def reindex_groups(self):
        """
        Reindex/rebuild the group coordinates and multinded in the dataset to match the original coordinates
        """
        return reindex_groups(self._obj)
    def rename_groups(self,grpmap):
        return rename_groups(self._obj,grpmap)

    @staticmethod
    def neqzeros(rhsdims,coords,lower=0):
        return neqzeros(rhsdims=rhsdims,coords=coords,lower=lower)
