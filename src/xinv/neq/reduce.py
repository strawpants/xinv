## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl


from xinv.core.attrs import find_xinv_unk_coord,find_neq_components,islower,xinv_st,get_state,get_type,xinv_tp,find_xinv_group_coords,get_xunk_size_coname

from xinv.core.tools import find_overlap_coords,find_ilocs,find_unk_idx,find_unk_idxv2
import numpy as np
import xarray as xr
from xinv.core.logging import xinvlogger
from xinv.linalg.inplace import cholesky_inplace,dtrsm_inplace,dsyrk_inplace
from xinv.neq import zeros as neqzeros


def ireduce(dsneq,idx,keep=False):
    """
        Reduce Normal equation system using indexing
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system to be reduced
        idx: list or np.ndarray 
            index of the coordinates to reduce or keep in the system 

    """
    
    u_sz,unkdim=get_xunk_size_coname(dsneq)
    #compute the complementary index
    if keep:
        idxkeep=idx
        if idxkeep.dtype == bool:
            idxreduce=~idxkeep
        else:
            idxreduce=~np.isin(np.arange(u_sz),idxkeep)
    else:
        idxreduce=idx
        if idxreduce.dtype == bool:
            idxkeep=~idxreduce
        else:
            idxkeep=~np.isin(np.arange(u_sz),idxreduce)

    #pointers to the input normal equation system
    i_N,i_rhs,i_x0,i_ltpl,i_sigma0,i_nobs,i_npara=find_neq_components(dsneq)

    #
    
    if i_x0 is None:
        i_x0=xr.zeros_like(i_rhs)
    
    
    i_lower=islower(i_N)

    #copy the upper/lower triangular (needed to get correct cross-section matrices)
    if i_lower:
        i_N+=np.tril(i_N,-1).T
    else:
        i_N+=np.triu(i_N,1).T

    #create new coordinates for the output system
    outcoords={}
    for k,v in dsneq.coords.items():
        if k == unkdim:
            outcoords[k]=v[idxkeep]
        elif  k in dsneq.get_index(unkdim).names:

            #ignore these as they will already be included in the multiIndex unkdim coordinate
            continue
        else:
            outcoords[k]=v
    
    o_dsneq= neqzeros(rhsdims=i_rhs.dims,coords=outcoords,lower=i_lower)
    

    #pointers to the new normal equation system
    o_N,o_rhs,o_x0,o_ltpl,o_sigma0,o_nobs,o_npara=find_neq_components(o_dsneq)
    
    #copy the relevant data from the input system
    o_x0[()]=i_x0[{unkdim:idxkeep}]
    o_ltpl[()]=i_ltpl
    o_sigma0[()]=i_sigma0
    o_nobs[()]=i_nobs
    o_npara[()]=i_npara

    #initialize the rhs with copy of the input
    o_rhs[()]=i_rhs[{unkdim:idxkeep}]

    #initialize the Nkk with copy of the input
    o_N[()]=i_N[{unkdim:idxkeep,unkdim+'_':idxkeep}]

    #obtain the sections of the input normal system
    Nrr=i_N[{unkdim:idxreduce,unkdim+'_':idxreduce}]
    
    Nrk=i_N[{unkdim:idxreduce,unkdim+'_':idxkeep}]
    

    #retrieve the Cholesky decomposition of the to be reduced part
    
    cholesky_inplace(Nrr)
    
    #Decorrelate the Nrk matrix with Nrr cholesky part
    # that is solve Nrr' *X=Nrk  for X
    dtrsm_inplace(Nrr,Nrk,trans=1)
    
    
    #decorrelate the rhs part
    br=i_rhs[{unkdim:idxreduce}]
    dtrsm_inplace(Nrr,br,trans=1)
    
    o_ltpl[()]-=br.dot(br,dim=unkdim)
    #update right hand side
    o_rhs[()]-=Nrk.T.dot(br,dim=unkdim).rename({unkdim+'_':unkdim})
    
    #Update the normal matrix
    dsyrk_inplace(o_N,Nrk,trans=1,beta=1.0,alpha=-1.0)

    return o_dsneq

def reduce(dsneq,labels=None, keep=False,**kwargs):
    """
        Reduce Normal equation system using coordinate labelling
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system to be reduced
        labels: array like
            search for these labels to reduce/keep in the default unknown coordinate
        **kwargs: 
            keyword arguments with the dimension name as key and a list of coordinate labels to be reduced from the system
            coord1 = reducelabels1 , .. coord2 = reducelabels2 

    """
    
    idxfound=find_unk_idxv2(dsneq,selargs=labels,**kwargs)
    return ireduce(dsneq,idx=idxfound,keep=keep)
  

def groupreduce(dsneq,groups=None,keep=False,**kwargs):
    """
    Reduce by groupname a group of parameters from a normal equation system
    Parameters
    ----------
    dsneq : xr.Dataset
        Dataset containing the normal equation system to be reduced
    groups : str, or dict
        The groupname of the parameters to be reduced, or a dictonary with criteria linked to the groupi. Note criteria are applied inclusive so all matches will be considered
    keep : bool, optional
        If True, the found group parameters are kept instead of reduced. The default is False.
    
    """
    
    if groups is not None and len(kwargs) != 0:
        raise ValueError("Cannot use both **kwargs and group argument at the same time")
    
    if len(kwargs) >0:
        groups=kwargs

    if type(groups) is str:
        #shortcut when a single groupname is suplied directly
        return reduce(dsneq,{groups:slice(None)},level_default=None,keep=keep)


    for ky,val in groups.items():
        #append a None to each group before submmitting it to reduce so we make the criteria inclusive all group criteria will be considered
        valnew=np.empty(len(val)+1,dtype='O')
        valnew[:-1]=val #noteL this leaves the last alue to None 
        groups[ky]=valnew

    return reduce(dsneq,labels=groups,keep=keep)
