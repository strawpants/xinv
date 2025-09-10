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
<<<<<<< HEAD
    if coord1.dtype != coord2.dtype:
        raise ValueError(f"Coordinates must have the same types, supplied are coord1: {coord1.dtype},coord2: {coord2.dtype}")

=======
    
>>>>>>> 66694e0 (Added transform function, updated fwdoperator interfacing, and more)
    intersect=np.intersect1d(coord1.data,coord2.data,return_indices=False)
    uniq1=np.setdiff1d(coord1, coord2, assume_unique=False)
    uniq2=np.setdiff1d(coord2, coord1, assume_unique=False)

    return uniq1, intersect, uniq2
