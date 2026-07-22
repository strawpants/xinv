#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.fwdrep import FwdRepOp
from xinv.fwd.polynomial import Polynomial
from fixtures import neqbase


     


@pytest.mark.parametrize("neqbase",["simple"],indirect=True)        
def test_rep(neqbase):
    """
    Test a repeat forward operator, allowing the expansion of parameters using the same forward operator
    Parameters
    ----------
    neqbase : xr.Dataset containing a normal equation system to apply the repeated operation 

    """
    
    ydat=np.arange(0,10)
    y0=4.5
    delta_y=1
    npoly=1
    
    polyfwd=Polynomial(n=npoly,poly_x='y',x0=y0,delta_x=delta_y,cache=False,unknown_dim='polyy')
    fwdrep=FwdRepOp(polyfwd,rep_dim='poly') 
    
    dstrans=None
    for y in ydat:
        #transform normal equation 
        if dstrans is None:
            dstrans=neqbase.xi.transform(fwdrep,y=[y])
        else:
            dstrans=dstrans.xi.add(neqbase.xi.transform(fwdrep,y=[y]))

    dssol_rep=dstrans.xi.solve()
    dssol=neqbase.xi.solve()
    
    #Since we didn't add any actual variation over y we expect the degree 0 polynomial in y to resolve to the original poly=[0,..] values
    for ipoly in dssol.poly.data:
        assert np.allclose(dssol.solution.sel(poly=[ipoly]),dssol_rep.solution.sel(polyy=[0],poly_rep=[ipoly]))

    # since we didn't intriduce any variation as a function of y we expect the trend (polyy=1) to be close to zero)
    assert np.allclose(dssol_rep.solution.sel(polyy=[1]),0)
    

