## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr

from xinv.neq import solve as neqsolve
from xinv.neq import transform as neqtransform
from xinv.neq import reduce,ireduce,groupreduce
from xinv.neq import regadd,regSys
from xinv.neq import fix,ifix,groupfix
from xinv.neq import set_x0 as neqset_x0
from xinv.neq import neqadd
from xinv.neq import zeros as neqzeros
from xinv.neq import BlockBy
from xinv.neq.build import build_normal as neqbuild_normal
from xinv.core.tools import select,find_ilocs2
from xinv.core.grouping import get_group,reindex_groups,rename_groups

from xinv.io.serialize import serialize,deserialize
from xinv.core.attrs import find_xinv_coords,xinv_tp,xinv_st,find_components,xunk_coords_attrs,find_component
import numpy as np

@xr.register_dataarray_accessor("xi")
class InverseDaAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    def build_normal(self,fwdop,ecov=1,**kwargs):
        """Builds a normal equation system from forward operators, data and an accompanying covariance"""
        
        #create the normal equation system
        return neqbuild_normal(fwdop,self._obj,ecov=ecov,**kwargs)
    

    def deepcopy(self,order='F'):
        daout=self._obj.copy(deep=True)
        if type(daout.data) == np.ndarray:
            daout.data=self._obj.data.copy(order=order)
        return daout

@xr.register_dataset_accessor("xi")
class InverseDsAccessor:
    """
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
   
    def find_xinv_type(self,xitype):
        try:
            return find_component(self._obj,xitype)
        except:
            return None


    @property
    def N(self):
        """
          Returns the current normal matrix variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.N)
    
    @property
    def COV(self):
        """
          Returns the current Covariance  matrix variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.COV)
    
    @property
    def rhs(self):
        """
          Returns the current right hand side variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.rhs)
    
    @property
    def x0(self):
        """
          Returns the current right hand side variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.x0)
    
    
    @property
    def nobs(self):
        """
          Returns the current apriori  variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.nobs)

    @property
    def npara(self):
        """
          Returns the current apriori  variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.npara)
    
    @property
    def ltpl(self):
        """
          Returns the current apriori  variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.ltpl)

    @property
    def sigma0(self):
        """
          Returns the current apriori  variable if it exists (None) otherwise
        """
        return self.find_xinv_type(xinv_tp.sigma0)

    
    @property
    def unk_co(self):
        """
          Returns the current i unknown coordinate variable
        """
        xunk_co=find_xinv_coords(self._obj,include=[xinv_tp.unk_co],state=xinv_st.linked)
        if len(xunk_co)!=1:
            raise ValueError("No or ambiguous linked unknown coordinate found")
        
        return next(iter(xunk_co.values()))


    @property
    def index(self):
        """
        Returns
        """
        unkdim,_=self.unknown_dim()
        return self._obj.get_index(unkdim)

    def sel(self,**kwargs):
        """
            Xarray like select but also applied to the transpose dimension of a normal equation system

        """
        return select(self._obj,**kwargs)

    def transform(self,fwdop,**kwargs):
        return neqtransform(self._obj,fwdop,**kwargs) #transform the normal equation system using a forward operator
    
    def solve(self,inplace=False):
        return neqsolve(self._obj,inplace) #solve the normal equation system
    
    def reduce(self,labels=None,keep=False,**kwargs):
        
        return reduce(self._obj,labels=labels,keep=keep,**kwargs) #reduce  parameters from the normal equation system
    def ireduce(self,idx,keep=False):
        return ireduce(self._obj,idx=idx,keep=keep) #reduce  parameters by index

    def reduce_groups(self,groups=None,keep=False,**kwargs):
        return groupreduce(self._obj,groups=groups,keep=keep,**kwargs)
    
    def fix(self,labels=None,keep=False,**kwargs):
        return fix(self._obj,labels=labels,keep=keep,**kwargs) #remove parameters from the normal equation system (fix them to their current apriori values)
    
    def ifix(self,idx,keep=False):
        """fix by index"""
        return ifix(self._obj,idx=idx,keep=keep) #remove parameters from the normal equation system (fix them to their current apriori values)
    
    def fix_groups(self,groups=None,keep=False,**kwargs):
        return groupfix(self._obj,groups=groups,keep=keep,**kwargs)

    def set_x0(self,dax0,is_delta=False,inplace=False):
        return neqset_x0(self._obj,dax0,is_delta,inplace) #change apriori values

    def add(self,dsneqother):
        return neqadd(self._obj,dsneqother) #add/merge another normal equation system

    def reg(self,dsreg,alpha=None,inplace=False):
        """
        Add a regularization to a normal equationsystem
        
        Parameters
        ----------
        dsreg: xr.Dataset or xr.DataArray
            Regularization system or matrix
        alpha:float,optional,
            apply a different scaling than provided with the data
        """
        if type(dsreg) == xr.DataArray:
            return regadd(self._obj,regSys(dsreg),alpha=alpha,inplace=inplace)
        else:
            return regadd(self._obj,dsreg,alpha=alpha,inplace=inplace)


    def get_group(self,level_name):
        return get_group(self._obj,level_name)
    
    def get_indexer(self,other):
        """
        return an indexer allowing the lookup of the values of other in the source dataset
        """
        xunk_co=find_xinv_coords(other,include=[xinv_tp.unk_co],state=xinv_st.linked)
        if len(xunk_co)!=1:
            raise ValueError("No or ambiguous linked unknown coordinate found in other")
        unkdim= next(iter(xunk_co.values())).dims[0]
        return find_ilocs2(self.index,other.get_index(unkdim))


    def serialize(self):
        return serialize(self._obj)
    
    def deserialize(self):
        return deserialize(self._obj)
    
    def rename_groups(self,renamemap=None,**kwargs):
        return rename_groups(self._obj,renamemap,**kwargs)

    @staticmethod
    def neqzeros(rhsdims,coords,lower=0):
        return neqzeros(rhsdims=rhsdims,coords=coords,lower=lower)

    def unknown_dim(self):
        
        """
        Convenience function to retrieve the name of the currently linked unknown coordinate dimension, and its transpose
        
        Returns 
        -------
        str
            The name of the currently linked unknown coordinate dimension
        
        """
        xunk_co=find_xinv_coords(self._obj,include=[xinv_tp.unk_co],state=xinv_st.linked)
        if len(xunk_co)!=1:
            breakpoint()
            raise ValueError("No or ambiguous linked unknown coordinate found")
        unkdim= next(iter(xunk_co.values())).dims[0]
        if unkdim+"_" in self._obj.dims:
            unkdim_=unkdim+"_"
        else:
            #return None if no transpose variant was found
            unkdim_=None
        return unkdim,unkdim_

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
    
    def trace(self):
        """
            Return the traces taking into account the sigma0 or alpha's of the matrix
        """
         

        matcomp=[xinv_tp.N,xinv_tp.COV,xinv_tp.REG]
        scalecomp=[xinv_tp.sigma0,xinv_tp.alpha]
        
        
        matfound=[(ky,val) for ky,val in zip(matcomp,find_components(self._obj,matcomp)) if val is not None]
        scalefound=[(ky,val) for ky,val in zip(scalecomp,find_components(self._obj,scalecomp)) if val is not None]
        
        
        if len(matfound) != 1 or len(scalefound) != 1:
            raise RuntimeError("Ambigious matrices or scales found")

        tpm,mat=matfound[0]
        tpsc,scale=scalefound[0]

        trc=np.diagonal(mat.data).sum()

        if tpm == xinv_tp.N:
            if tpsc != xinv_tp.sigma0:
                raise RuntimeError("Expecting sigma0 with Normal matrix")
            trc/=np.power(scale,2)
        elif tpm == xinv_tp.COV:
            if tpsc != xinv_tp.sigma0:
                raise RuntimeError("Expecting sigma0 with Covariance matrix")
            trc*=np.power(scale,2)
        elif tpm == xinv_tp.REG:
            if tpsc != xinv_tp.alpha:
                raise RuntimeError("Expecting alpha with Regularization matrix")
            trc*=scale

        return trc 
    
    def blockby(self,level_name):
        return BlockBy(self._obj,level_name)

    def deepcopy(self,order='F'):
        """
            Make a truly deep copy of the input (xarray does not always make a truly deep copf the data
        """
        dsout=self._obj.copy(deep=True)
        #loop to explicitly copy numpy ndarrays
        for vname,var in self._obj.data_vars.items():
            if type(var.data) == np.ndarray:
                dsout[vname].data=var.data.copy(order=order)
        return dsout


