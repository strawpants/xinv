## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from xinv.core.attrs import find_xinv_unk_coord,get_state,get_type,xinv_tp,xinv_st,find_xinv_group_coords
from xinv.core.logging import xinvlogger

def find_ilocs(dsneq,dim,elements):
    idxname=f"_{dim}_idx"
    if idxname not in dsneq.keys():
        dsneq[idxname]=(dim,np.arange(dsneq.sizes[dim]))
    return dsneq[idxname].loc[elements].data

def find_overlap_coords(coord1,coord2):
    """
        Find the unique and common coordinates between two xarray coordinates
    """
    unkdim1=coord1.dims[0]
    unkdim2=coord2.dims[0]

    if unkdim1 != unkdim2:
        raise ValueError("Coordinates must have the same dimension name")
    if coord1.dtype != coord2.dtype:
        raise ValueError(f"Coordinates must have the same types, supplied are coord1: {coord1.dtype},coord2: {coord2.dtype}")

    intersect=np.intersect1d(coord1.data,coord2.data,return_indices=False)
    uniq1=np.setdiff1d(coord1, coord2, assume_unique=False)
    uniq2=np.setdiff1d(coord2, coord1, assume_unique=False)

    return uniq1, intersect, uniq2


def find_unk_idx(dsneq,sort=True,**kwargs):
    """ 
        Find the indices of a set of unknown parameters in the unknown vector of a normal equation system 
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system
        sort: bool, optional
            If True, the output indices are sorted in ascending order. The default is True.
        kwargs: dict
            keyword arguments with dimension names as keys and the elements to find as values
            e.g. poly=[0,1],harmonics_seasonal=[1,2]
        Returns:
        --------
        idxfound: np.ndarray or None
            Indices of the found unknown parameters in the unknown vector, or None if no parameters were found
        idxremaining: np.ndarray or None
            
            Indices of the unknown parameters that are complementary to the ones found
        idxnotfound: np.ndarray or None
            Indices of the requested unknown parameters that were not found in the system
            
    """

    xunk_co=find_xinv_unk_coord(dsneq)
    
    unkdim=xunk_co.dims[0]
    
    group_id_co=None
    group_seq_co=None

    notfound=[]
    found=[]
    remaining=[]
    for coname,searchparams in kwargs.items():
        co_search=dsneq[coname]
        if get_type(co_search) != xinv_tp.unk_co:
            raise ValueError(f"Missing xinv_type: supplied coordinate name {coname} has no valid relation with unknown coordinate {xunk_co.name}")
        dimname=co_search.dims[0]
        if searchparams is not type(xr.DataArray):
            #turn into DataArray
            searchparams=xr.DataArray(searchparams,dims=dimname)
        #find unique and overlapping coordinates over the unknown dimension
        notfnd,fnd,remng=find_overlap_coords(searchparams,co_search)
       
        if get_state(co_search) == xinv_st.unlinked:
            #we may have to apply an additional lookup in the group unknown multiindex
            if group_id_co is None and group_seq_co is None:
                group_id_co,group_seq_co=find_xinv_group_coords(dsneq)
            notfound.extend([(coname,i) for i in find_ilocs(dsneq,coname,notfnd)])
            found.extend([(coname,i) for i in find_ilocs(dsneq,coname,fnd)])
            remaining.extend([(coname,i) for i in find_ilocs(dsneq,coname,remng)])

        elif get_state(co_search) == xinv_st.linked:
            notfound.extend(notfnd)
            found.extend(fnd)
            remaining.extend(remng)
        else:
            raise ValueError(f"Reduction coordinate {coname} has no valid link state")
    
    

    #index vector of the found parameters
    idxfound=find_ilocs(dsneq,unkdim,found) if len(found) > 0 else None
    idxnotfound=find_ilocs(dsneq,unkdim,notfound) if len(notfound) > 0 else None
    idxremaining=find_ilocs(dsneq,unkdim,remaining) if len(remaining) > 0 else None
    if sort:
        if idxfound is not None:
            idxfound=np.sort(idxfound)
        if idxremaining is not None:
            idxremaining=np.sort(idxremaining)
        if idxnotfound is not None:
            idxnotfound=np.sort(idxnotfound)
    
    return idxfound,idxremaining,idxnotfound
