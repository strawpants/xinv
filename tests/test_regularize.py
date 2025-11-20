import pytest
import xarray as xr
import numpy as np
from xinv import *
import os
from xinv.core.tools import find_ilocs
from xinv.core.attrs import find_xinv_coords,xinv_tp,xinv_st,find_neq_components
from xinv.core.exceptions import XinvIllposedError
from xinv.neq.regularize import getTikhonov
from fixtures import neqbase

@pytest.mark.parametrize("neqbase",["ill posed"],indirect=True)
def test_tikhonov(neqbase):
    """
        Test a Tikhonov regularization on an ill-posed system     
    """
    

    try:
        neqbase.xi.solve()
        assert False
    except XinvIllposedError:
        # failed succesfully
        assert True 

    # Create a Tikhonov regularization matrix on a subset of the parameters
    dstik=getTikhonov(poly2=[0])

    
    
    try:
        dstik2=dstik.xi.as_group(poly2='xinv_unk')
        neqbasereg=neqbase.xi.reg(dstik2,alpha=1e3)
        assert False
    except:
        #iWe expect to land here because the systems are not consistent
        assert True
    
    #this one shoudl be fine
    neqbasereg=neqbase.xi.reg(dstik,alpha=1e3)


    #we should now be able to solve the system
    dssolv=neqbasereg.xi.solve()
    # Check if the results make sense (constraint values should be close to zero)
    assert np.allclose(dssolv.solution.sel(xinv_unk=('poly2',0)),0)
