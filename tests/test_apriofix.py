## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
import os

from fixtures import neqbase
from xinv.core.tools import find_ilocs,find_unk_idx
from xinv.core.logging import xinvlogger

@pytest.mark.parametrize("neqbase",["simple","stacked"],indirect=True)
@pytest.mark.parametrize("keep",[True])
def test_aprifix(neqbase,keep):
    """
    Test the apriori setting and removal of a set of unknown parameters from a neq system 
    """
    
    
    #compute the overall solution
    dssol=neqbase.xi.solve()

    #get the index vector corresponding to some polynomials to fix (or keep)

    fixpolyparam=[1,2]
    idxfix,idxremain,_=find_unk_idx(dssol,poly=fixpolyparam)

    unkdim=dssol.xi.unknown_dim()

    #set the apriori values of the unknown parameters to that from the overall solution
    if keep:
        dsneq_apriset=neqbase.xi.set_x0(dssol.solution[{unkdim:idxremain}])
    else:
        dsneq_apriset=neqbase.xi.set_x0(dssol.solution[{unkdim:idxfix}])
    

    dssolapr=dsneq_apriset.xi.solve()

    #check if x0+solution is the same as the overall solution


    #solution should be the same on the non-reduced neq system
    assert np.allclose(dssolapr.solution+dssolapr.x0,dssol.solution)

    assert np.allclose(dssolapr.COV,dssol.COV)
    assert np.allclose(dssolapr.ltpl,dssol.ltpl)
    assert np.allclose(dssolapr.sigma0,dssol.sigma0)

    # #fix the relevant parameters to their apriori values
    dsneq_aprifix=dsneq_apriset.xi.fix(poly=fixpolyparam,keep=keep)
    
    dssolfix=dsneq_aprifix.xi.solve()
    if keep:
        dssolsub=dssol.isel({unkdim:idxfix,unkdim+"_":idxfix})
    else:
        dssolsub=dssol.isel({unkdim:idxremain,unkdim+"_":idxremain})
    soldif=np.abs(dssolfix.solution+dssolfix.x0-dssolsub.solution)
    # stddev=xr.DataArray(np.sqrt(np.diag(dssolfix.COV)),dims=unkdim,coords={unkdim:dssolfix[unkdim]})*dssolfix.sigma0
    
    assert np.all(soldif < 1e-10)
    



    

@pytest.mark.parametrize("neqbase",["stacked"],indirect=True)
@pytest.mark.parametrize("keep",[False,True])
def test_groupfix(neqbase,keep):
    """
    Test the apriori fixing and removal of a  group of unknown parameters from a neq system 
    """
    

    #compute the overall solution
    idxpoly=neqbase.xinv_grp_id == "poly" 
    
    dssol=neqbase.xi.solve()
    
    unkdim=dssol.xi.unknown_dim()
    if keep:
        dsneq_apriset=neqbase.xi.set_x0(dssol.solution)
    else:
        dsneq_apriset=neqbase.xi.set_x0(dssol.solution[{unkdim:idxpoly}])

    neqbase_aprifix=dsneq_apriset.xi.groupfix('poly',keep=keep)


    npara=neqbase.sizes['poly']
    


    dssolapri=neqbase_aprifix.xi.solve()


    if keep:
        assert dssolapri.sizes[unkdim] == npara
    else:
        assert neqbase.sizes[unkdim]-dssolapri.sizes[unkdim] == npara
    if keep:
        soldif=np.abs(dssolapri.solution+dssolapri.x0-dssol.solution[idxpoly])
    else:
        soldif=np.abs(dssolapri.solution+dssolapri.x0-dssol.solution[~idxpoly])
    
    assert np.all(soldif < 1e-10)
