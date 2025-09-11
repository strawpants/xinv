## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr

from xinv.neq import solve as neqsolve
from xinv.neq import transform as neqtransform
from xinv.neq import groupreduce,reduce,ireduce

from xinv.neq import fix as neqfix
from xinv.neq import set_apriori as neqset_apriori
from xinv.neq import neqadd
from xinv.neq import zeros as neqzeros
from xinv.neq.build import build_normal as neqbuild_normal

from xinv.core.grouping import get_group,reindex_groups,rename_groups

from xinv.core.attrs import find_xinv_coords,xinv_tp,xinv_st

@xr.register_dataarray_accessor("xi")
class InverseDaAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def build_normal(self,fwdop,ecov=1,**kwargs):
        """Builds a normal equation system from forward operators, data and an accompanying covariance"""
        
        #create the normal equation system
        return neqbuild_normal(fwdop,self._obj,ecov=ecov,**kwargs)

@xr.register_dataset_accessor("xi")
class InverseDsAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def transform(self,fwdop,**kwargs):
        return neqtransform(self._obj,fwdop,**kwargs) #transform the normal equation system using a forward operator
    
    def solve(self,inplace=False):
        return neqsolve(self._obj,inplace) #solve the normal equation system
    
    def reduce(self,keep=False,**kwargs):
        return reduce(self._obj,keep=keep,**kwargs) #reduce  parameters from the normal equation system
    def ireduce(self,idx,keep=False):
        return ireduce(self._obj,idx=idx,keep=keep) #reduce  parameters by index
    
    def groupreduce(self,groupname,keep=False):
        return groupreduce(self._obj,groupname,keep=keep) #reduce  parameters by groupname index
    
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

    def unknown_dim(self):
        
        """
        Convenience function to retrieve the name of the currently linked unknown coordinate dimension
        
        Returns 
        -------
        str
            The name of the currently linked unknown coordinate dimension
        
        """
        xunk_co=find_xinv_coords(self._obj,include=[xinv_tp.unk_co],state=xinv_st.linked)
        if len(xunk_co)!=1:
            raise ValueError("No or ambiguous linked unknown coordinate found")
        return next(iter(xunk_co.values())).dims[0]

    def unknown_size(self):
        """
        Convenience function to retrieve the size of the currently linked unknown coordinate
        
        Returns 
        -------
        int
            The size of the currently linked unknown coordinate
        
        """
        xunk_co=find_xinv_coords(self._obj,include=[xinv_tp.unk_co],state=xinv_st.linked)
        if len(xunk_co)!=1:
            raise ValueError("No or ambiguous linked unknown coordinate found")
        return self._obj.sizes[next(iter(xunk_co.values())).dims[0]]
