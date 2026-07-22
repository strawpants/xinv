## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from xinv.core.attrs import find_xinv_unk_coord,get_state,get_type,xinv_tp,xinv_st,xunk_coords_attrs
from xinv.core.logging import xinvlogger
import pandas as pd
from warnings import deprecated

@deprecated("Use find_ilocs2 instead")
def find_ilocs(dsneq,dim,elements,reverse=False):

    idxsrc=dsneq.get_index(dim)
    if hasattr(idxsrc,'to_flat_index'):
        idxsrc=idxsrc.to_flat_index()

    idx=idxsrc.get_indexer(elements)
    
    if reverse:
        idx=~pd.RangeIndex(idxsrc.size).isin(idx)

    return idx

def find_ilocs2(idxsrc,elements,inverse=False):
    """
        get the location of the elements within idxsrc 
    """
    if type(idxsrc) == pd.MultiIndex:
    #elements needs to be made consistent
        if type(elements) == pd.MultiIndex or type(elements)  == pd.Index:
            if elements.names != idxsrc.names:
                #expand elements into a compatible MultiIndex 
                dfel=elements.to_frame()
                for name in idxsrc.names:
                    if name not in elements.names:
                        dfel.insert(0,name,np.nan)

                dfel.set_index(idxsrc.names,inplace=True)
                elements=dfel.index
            if hasattr(elements,'to_flat_index'):
                elements=elements.to_flat_index()

        idx=idxsrc.to_flat_index().get_indexer(elements)

    elif type(idxsrc) == pd.Index:
        #just directly figure out the indexer
        idx=idxsrc.get_indexer(elements)
    else:
        raise ValueError("idxsrc needs to be a pandas index")



    if inverse:
        idx=~pd.RangeIndex(idxsrc.size).isin(idx)
    
    return idx 


def unique_union(idx1,idx2):
    """
        Find the unique union of 2 indicesa, while also considering level names
    """
    unionlev=np.union1d(idx1.names,idx2.names)
    if len(unionlev) == 1:
        #result will be ai simple index
        idxout=idx1.union(idx2,sort=False).unique()
        idxout.set_names(idx1.names,inplace=True)
        
    elif len(unionlev) == idx1.nlevels and len(unionlev) == idx2.nlevels:
        #result will be the same for as the input
        idxout=idx1.union(idx2.to_flat_index(),sort=False).unique()
        idxout.set_names(idx1.names,inplace=True)
    else:
        #result will have additional levels
        idxout=pd.concat([idx1.to_frame(),idx2.to_frame()]).set_index(list(unionlev)).index.unique()

    
    return idxout

def find_overlap(idx1,idx2):

    if idx1.name != idx2.name:
        raise ValueError("Indexes must have the same name")
    if idx1.dtype != idx2.dtype:
        raise ValueError(f"Indexes must have the same types, supplied are coord1: {idx1.dtype},coord2: {idx2.dtype}")

    intersect=idx1.intersection(idx2)
    uniq1=idx1.difference(idx2)
    uniq2=idx2.difference(idx1)
    return uniq1,intersect,uniq2

@deprecated("Use find_overlap instead")
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

def find_unk_idxv2(dsneq,selargs=None,sort=True,level_default=slice(None),force_boolean=False,inverse=False,**kwargs):
    """ 
        Find the indices of a set of unknown parameters in the unknown vector of a normal equation system 
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system
        selargs: array like, tuple, or dictionary like xarray's sel() selection call
            List/array/dict of parameters to search for in the default unknown coordinate
        sort: bool, optional
            If True, the output indices are sorted in ascending order. The default is True.
        level_default:
            use this values as the default for when it is not explicitly specified (default is slice(None) which means it is not used as a search restriction)
        kwargs: dict
            Convenience keyword arguments with multiIndex level names and sequences to search for
            e.g. poly=[0,1],harmonics_seasonal=[1,2]
        Returns:
        --------
        idxfound: np.ndarray
            Indices obeying the selection criteria, None when nothing is found
    """
    
    if (len(kwargs) >= 1 and selargs is not None) :
        raise ValueError("find_unk_idx2: cannot use both selargs and named arguments at the same time")
    elif selargs is None and len(kwargs) ==0:
        raise ValueError("find_unk_idx2: must provide either selargs= or parameters inamed after the multindex levels")

    xunk_co=find_xinv_unk_coord(dsneq)
    unkdim=xunk_co.name
    xunk_idx=xunk_co.indexes[unkdim]
    if selargs is not None:
        if type(selargs) == dict:
            #just copy into kwargs and handle later
            kwargs=selargs
        else:
            #treat as the selection on the found unknown coordinate
            kwargs[unkdim]=selargs
    
    if kwargs:
        if unkdim in kwargs:
            if len(kwargs) != 1:
                raise ValueError("Cannot mix multiIndex top name and sublevels")
            #just use as isain get_locs
            locarg=kwargs[unkdim]
        else:
            #construct the argument for get_locs
            lookup={ky:i for i,ky in enumerate(xunk_idx.names)} 
            locarg=[level_default for i in range(xunk_idx.nlevels)]
             
            for name,seq in kwargs.items():
                try:
                    locarg[lookup[name]]=seq
                except KeyError:
                    raise ValueError(f"{name} not found as a level of the {unkdim} Index")

    
    try:
        if hasattr(xunk_idx,'get_locs'):
            #treart as multiindex
            idx=xunk_idx.get_locs(locarg)
        else:
            idx=xunk_idx.isin(locarg)
    except KeyError :
        raise ValueError(f"Cannot find all values in {unkdim}")
    if sort and idx.dtype != bool:
        idx=np.sort(idx)
    if inverse:
        idx=~pd.RangeIndex(xunk_idx.size).isin(idx)

    if force_boolean and idx.dtype != bool:
        idx=pd.RangeIndex(xunk_idx.size).isin(idx)
    return idx

@deprecated("This routine will be phased out due to different grouping treatment")
def find_unk_idx(dsneq,labels=None,sort=True,**kwargs):
    """ 
        Find the indices of a set of unknown parameters in the unknown vector of a normal equation system 
        Parameters:
        -----------
        dsneq: xarray.Dataset
            Dataset containing the normal equation system
        labels: array like
            List/array of parameters to search for in the default unknown coordinate
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
        notfound: int or None
            Amount of parameters not found in the system
            
    """
    xunk_co=find_xinv_unk_coord(dsneq)
    
    unkdim=xunk_co.dims[0]
   
    if labels is not None:
        #add unnamed labels to search for to the unknown coordinate
        kwargs[unkdim]=labels

    group_id_co=None
    group_seq_co=None

    notfound=0
    found=[]
    remaining=[]
    for coname,searchparams in kwargs.items():
        co_search=dsneq[coname]
        if get_type(co_search) != xinv_tp.unk_co:
            raise ValueError(f"Missing xinv_type: supplied coordinate name {coname} has no valid relation with unknown coordinate {xunk_co.name}")
        dimname=co_search.dims[0]
        if searchparams is not type(xr.DataArray):
            #turn into DataArray

            if unkdim in co_search.indexes and type(co_search.indexes[unkdim]) == pd.MultiIndex:
                searchparams=xr.Coordinates.from_pandas_multiindex(pd.MultiIndex.from_tuples(searchparams,names=co_search.indexes[unkdim].names),unkdim).to_dataset()[unkdim]
            else:
                #okay just turn the values in a DataArray
                searchparams=xr.DataArray(searchparams,dims=dimname)
        #find unique and overlapping coordinates over the unknown dimension
        notfnd,fnd,remng=find_overlap_coords(searchparams,co_search)
        if get_state(co_search) == xinv_st.unlinked:
            #we may have to apply an additional lookup in the group unknown multiindex
            if group_id_co is None and group_seq_co is None:
                group_id_co,group_seq_co=find_xinv_group_coords(dsneq)
            found.extend([(coname,i) for i in find_ilocs(dsneq,coname,fnd)])

        elif get_state(co_search) == xinv_st.linked:
            found.extend(fnd)
        else:
            raise ValueError(f"Reduction coordinate {coname} has no valid link state")
        notfound+=len(notfnd)
    #figure out remaining parameters

    #index vector of the found parameters
    idxfound=find_ilocs(dsneq,unkdim,found) if len(found) > 0 else None
    
    idxremaining=find_ilocs(dsneq,unkdim,found,reverse=True) 

    

    if sort:
        if idxfound is not None:
            idxfound=np.sort(idxfound)
        if idxremaining is not None:
            idxremaining=np.sort(idxremaining)
    
    return idxfound,idxremaining,notfound


def select(dsin,**kwargs):
    """
        Select parts of Normal equation system/Error covariance system obeying certain selct criteria
    """
    
    idx_select=find_unk_idxv2(dsin,**kwargs)
     
    unkdim,unkdim_=dsin.xi.unknown_dim() 
    if unkdim_ is not None:
        dsout=dsin.isel({unkdim:idx_select,unkdim_:idx_select})
    else:
        dsout=dsin.isel({unkdim:idx_select})
    
    #set attributes

    dsout[unkdim].attrs.update(xunk_coords_attrs(state=xinv_st.linked))
    

    return dsout
