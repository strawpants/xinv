## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl


from xinv.core.attrs import find_component,xinv_tp,find_xinv_group_coords,get_xunk_size_coname

from xinv.core.tools import find_unk_idx,find_unk_idxv2
import numpy as np
from xinv.core.logging import xinvlogger


def ifix(dsneq,idx,keep=False):
    """
        Fix/remove parameters in a normal equation system to their apriori values using indexing
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system to be fixed
        idx: list or np.ndarray 
            index of the coordinates to fix or keep in the system 

    """
    
    u_sz,unkdim=get_xunk_size_coname(dsneq)
    #compute the complementary index
    if keep:
        idxkeep=idx
        if idxkeep.dtype == bool:
            idxfix=~idxkeep
        else:
            idxfix=~np.isin(np.arange(u_sz),idxkeep)
    else:
        idxfix=idx
        if idxfix.dtype == bool:
            idxkeep=~idxfix
        else:
            idxkeep=~np.isin(np.arange(u_sz),idxfix)

    o_dsneq=dsneq.isel({unkdim:idxkeep,unkdim+'_':idxkeep})
    io_npara=find_component(o_dsneq,xinv_tp.npara)
    #update amount of unknown parameters
    io_npara[()]-=len(idxfix)

    return o_dsneq


def fix(dsneq, labels=None, keep=False,**kwargs):
    """
        Fix/remove parameters from a Normal equation system using coordinate labelling
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system to be fixed
        labels: array like
            search for these labels to fix/keep in the default unknown coordinate
        keep: bool, optional
            If True, the parameters are kept instead of fixed. The default is False.
        **kwargs: 
            keyword arguments with the dimension name as key and a list of coordinate labels to be fixed/removed from the system
            coord1 = fixlabels1 , .. coord2 = fixlabels2 

    """
    idxfound=find_unk_idxv2(dsneq,selargs=labels,**kwargs)
    return ifix(dsneq,idx=idxfound,keep=keep)

def groupfix(dsneq,groups=None,keep=False,**kwargs):
    """
    Fix by groupname a group of parameters from a normal equation system
    Parameters
    ----------
    dsneq : xr.Dataset
        Dataset containing the normal equation system to be reduced
    groups : str, or dict
        The groupname of the parameters to be fixed, or a dictonary with criteria linked to the groupi. Note criteria are applied inclusive so all matches will be considered
    keep : bool, optional
        If True, the found group parameters are kept instead of fixed. The default is False.
    
    """
    
    if groups is not None and len(kwargs) != 0:
        raise ValueError("Cannot use both **kwargs and group argument at the same time")
    
    if len(kwargs) >0:
        groups=kwargs

    if type(groups) is str:
        #shortcut when a single groupname is suplied directly
        return fix(dsneq,{groups:slice(None)},level_default=None,keep=keep)


    for ky,val in groups.items():
        #append a None to each group before submmitting it to reduce so we make the criteria inclusive all group criteria will be considered
        valnew=np.empty(len(val)+1,dtype='O')
        valnew[:-1]=val #noteL this leaves the last alue to None 
        groups[ky]=valnew

    return fix(dsneq,labels=groups,keep=keep)
