#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.fwdrep import FwdRepOp
from xinv.fwd.polynomial import Polynomial
from xinv.core.exceptions import XinvIllposedError
import os
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
    
    polyfwd=Polynomial(n=npoly,poly_x='y',x0=y0,delta_x=delta_y,cache=False,unknown_dim='poly2')
    fwdrep=FwdRepOp(polyfwd,rep_dim='poly') 
    
    for y in ydat:
        #transform normal equation 
        dstrans=neqbase.xi.transform(fwdrep,y=[y])
        
        dstrans.xi.fix(keep=True,poly2=1)

        assert False

