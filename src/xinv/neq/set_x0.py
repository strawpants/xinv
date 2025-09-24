## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from xinv.core.attrs import find_neq_components,x0_attrs
from xinv.core.tools import find_ilocs
from xinv.linalg.inplace import dsymm_inplace

def set_x0(dsneq:xr.Dataset, daapri:xr.DataArray, apriori_is_delta=False, inplace=False):
    """ Change apriori values in a normal equation system """

    if not inplace:
    #copy entire NEQ and operate on that one in place
        dsneq=dsneq.copy(deep=True)

    #note prefix io_ denotes used for in and output, i_ is used for input only
    i_N,io_rhs,io_x0,io_ltpl,_,_,_=find_neq_components(dsneq)
    
    unkdim=dsneq.xi.unknown_dim()

    if io_x0 is None:
        #create a new  x0 entry
        dsneq['x0']=xr.zeros_like(io_rhs.reset_index(unkdim))
        dsneq.x0.attrs=x0_attrs()
        io_x0=dsneq.x0

    #find the indices of the apriori values in the unknown vector
    idxapri=find_ilocs(dsneq,unkdim,daapri.coords[unkdim])
    
    deltax0=xr.zeros_like(io_rhs.reset_index(unkdim))
    aslc={unkdim:idxapri}
    if apriori_is_delta:
        # The apriori input is used as an increment to the existing x0
        deltax0[aslc]=daapri
        io_x0+=deltax0
    else:
        # The input is the new target apriori value and the deltax needed is computed from the existing x0
        deltax0[aslc]=daapri-io_x0[aslc]
        io_x0[aslc]=daapri
    
    
    # (1) first update to ltpl
    io_ltpl[()]-=deltax0.dot(io_rhs,dim=unkdim)
    

    # (2) update right hand side with symmetric matrix multiplication
    dsymm_inplace(A=i_N,B=deltax0,C=io_rhs,alpha=-1.0,beta=1.0)
    
    # (3) update ltpl again
    io_ltpl[()]-=deltax0.dot(io_rhs,dim=unkdim)
    

    return dsneq
