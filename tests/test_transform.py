import pytest
import xarray as xr
import numpy as np
from xinv import *
import os
from xinv.core.tools import find_ilocs
from xinv.core.attrs import find_xinv_coords,xinv_tp,xinv_st,find_neq_components
from xinv.neq import transform

@pytest.fixture
def neqbase(request):
    """
    load a normal equation system from file 
    """
    if request.param=='simple':
        #simple normal equation system
        neqfile1=os.path.join(os.path.dirname(__file__),f'testdata/neqpoly_simple.nc')
        dsneq=xr.open_dataset(neqfile1)
    else:
        neqfile1=os.path.join(os.path.dirname(__file__),f'testdata/neqpoly.nc')
        dsneq=xr.open_dataset(neqfile1).xi.reindex_groups()
    
    return dsneq

@pytest.mark.parametrize("neqbase",["simple","stacked"],indirect=True)
def test_identity_transform(neqbase):

    N,rhs,ltpl,sigma0,nobs,npara=find_neq_components(neqbase)
    unkdim=N.dims[0]

    B=np.eye(neqbase.sizes[unkdim])
    fwdoperator=xr.DataArray(B,dims=["unkdim_", unkdim],name="B")


    ds_trans=transform(neqbase,fwdoperator)

    Nout,rhsout,ltplout,sigma0out,nobsout,nparaout=find_neq_components(ds_trans)

    assert np.allclose(Nout.data,N.data,atol=1e-15)
    assert np.allclose(rhsout.data,rhs.data, atol=1e-15)


