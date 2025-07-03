## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr

def set_apriori(dsneq:xr.Dataset, daapri:xr.DataArray, absolute=False, inplace=False):
    
    """ Change apriori values in a normal equation system """

    if not inplace:
    #copy entire NEQ and operate on that one in place
        dsneq=dsneq.copy(deep=True)

    N,rhs,ltpl,_,_,_=find_neq_components(dsneq)

    try:
        x0=find_component(dsneq,xinv_tp.x0)
    except:
        # it may not be there
        x0=xr.zeros_like(rhs).assign_attrs(x0_attrs())

    
    if absolute:
        ## take the differences
        deltax0=daapri-x0
        x0_new=daapri
        x0[()]=x0_new

    else:
        # compute the deltax0 internally
        deltax0=daapri
        x0_new=x0+deltax0        
        x0[()]=x0_new


    ## update ltpl using the a priori in place
    ltpl[()]+= (-2*deltax0.transpose().data@rhs.data) + (deltax0.transpose().data@N.data@deltax0.data)

    ## update rhs using the a priori in place
    rhs[()]-=N.data.dot(deltax0.data)
    
    
    if inplace:
        return None
    else:
        return dsneq