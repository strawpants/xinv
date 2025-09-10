## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr

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
    
    intersect=np.intersect1d(coord1.data,coord2.data,return_indices=False)
    uniq1=np.setdiff1d(coord1, coord2, assume_unique=False)
    uniq2=np.setdiff1d(coord2, coord1, assume_unique=False)

    return uniq1, intersect, uniq2
