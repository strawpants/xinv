## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl


from xinv.core.attrs import find_component,xinv_tp,find_xinv_group_coords,get_xunk_size_coname

from xinv.core.tools import find_unk_idx
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

    o_dsneq=dsneq.sel({unkdim:idxkeep,unkdim+'_':idxkeep})
    io_npara=find_component(o_dsneq,xinv_tp.npara)
    #update amount of unknown parameters
    io_npara[()]-=len(idxfix)

    return o_dsneq


def fix(dsneq, keep=False,**kwargs):
    """
        Fix/remove parameters from a Normal equation system using coordinate labelling
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system to be fixed
        keep: bool, optional
            If True, the parameters are kept instead of fixed. The default is False.
        **kwargs: 
            keyword arguments with the dimension name as key and a list of coordinate labels to be fixed/removed from the system
            coord1 = fixlabels1 , .. coord2 = fixlabels2 

    """
    
    idxfound,idxremaining,idxnotfound=find_unk_idx(dsneq,**kwargs)
    if idxnotfound is not None:
        xinvlogger.warning(f"Fix parameters contain values {idxnotfound} which are not found in the input normal equation system, ignoring those")

    if (not keep and idxremaining is None) or (keep and idxfound is None):
        xinvlogger.warning("Nothing to fix, returning input")
        return dsneq
    elif (not keep and idxfound is None) or (keep and idxremaining is None):
        #cannot fix all unknowns
        raise ValueError("Fix parameters contain all unknown parameters, cannot fix all unknowns")
    
    return ifix(dsneq,idx=idxfound,keep=keep)

def groupfix(dsneq,groupname,keep=False):
    """
    Fix/remove by groupname a group of parameters from a normal equation system
    Parameters
    ----------
    dsneq : xr.Dataset
        Dataset containing the normal equation system to be fixed/removed
    groupname : str
        The groupname of the parameters to be fixed/removed
    keep : bool, optional
        If True, the group parameters are kept instead of fixed. The default is False.
    """


    #test whether the groupname actually exists
    grpid_co,grpseq_co=find_xinv_group_coords(dsneq)
    if grpid_co is None or grpseq_co is None:
        raise ValueError("No group coordinates found in the normal equation system")

    if groupname not in grpid_co.data:
        raise ValueError(f"Groupname {groupname} not found in the normal equation system")

    idx=grpid_co.data == groupname
    return ifix(dsneq,idx=idx,keep=keep)


