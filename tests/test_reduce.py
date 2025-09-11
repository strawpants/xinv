## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
import os

from fixtures import neqbase
from xinv.core.tools import find_ilocs


@pytest.mark.parametrize("neqbase",["simple","stacked"],indirect=True)
@pytest.mark.parametrize("keep",[True,False])
def test_reduce(neqbase,keep):
    """
    Test the reduction of a set of unknown parameters from a neq system 
    """
    reducepolyparam=[2]
    if "harmonics_seasonal" in neqbase.coords:
        neqbase_red=neqbase.xi.reduce(poly=reducepolyparam,harmonics_seasonal=[2],keep=keep)
        npara=len(reducepolyparam)+1
    else:
        neqbase_red=neqbase.xi.reduce(poly=reducepolyparam,keep=keep)
        npara=len(reducepolyparam)
    

    dssolred=neqbase_red.xi.solve()

    #check if the correct variables were removed kept

    dssol=neqbase.xi.solve()
    unkdim=neqbase.xi.unknown_dim()
    #select the same parameters from the non-reduced solution
    idxkeep=find_ilocs(dssol,unkdim,dssolred[unkdim])
    if keep:
        assert len(idxkeep) == npara
    else:
        assert neqbase.sizes[unkdim]-len(idxkeep) == npara

    dssolsub=dssol.isel({unkdim:idxkeep,unkdim+"_":idxkeep})

    #solution should be the same on the non-reduced neq system
    assert np.allclose(dssolred.solution,dssolsub.solution)

    assert np.allclose(dssolred.COV,dssolsub.COV)
    assert np.allclose(dssolred.ltpl,dssolsub.ltpl)
    assert np.allclose(dssolred.sigma0,dssolsub.sigma0)

    

@pytest.mark.parametrize("neqbase",["simple","stacked"],indirect=True)
@pytest.mark.parametrize("keep",[True,False])
def test_groupreduce(neqbase,keep):
    """
    Test the reduction of a  group of unknown parameters from a neq system 
    """
    try:
        neqbase_red=neqbase.xi.groupreduce('poly',keep=keep)

    except ValueError:
        #capture errors when there are not dedicated groups
        if "harmonics_seasonal" not in neqbase.coords:
            #ok just return
            assert True
            return


    npara=neqbase.sizes['poly']
    


    dssolred=neqbase_red.xi.solve()

    #check if the correct variables were removed kept

    dssol=neqbase.xi.solve()
    unkdim=neqbase.xi.unknown_dim()
    #select the same parameters from the non-reduced solution
    idxkeep=find_ilocs(dssol,unkdim,dssolred[unkdim])
    if keep:
        assert len(idxkeep) == npara
    else:
        assert neqbase.sizes[unkdim]-len(idxkeep) == npara

    dssolsub=dssol.isel({unkdim:idxkeep,unkdim+"_":idxkeep})

    #solution should be the same on the non-reduced neq system
    assert np.allclose(dssolred.solution,dssolsub.solution)

    assert np.allclose(dssolred.COV,dssolsub.COV)
    assert np.allclose(dssolred.ltpl,dssolsub.ltpl)
    assert np.allclose(dssolred.sigma0,dssolsub.sigma0)

