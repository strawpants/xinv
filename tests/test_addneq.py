#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
import os
from xinv.core.tools import find_ilocs
from xinv.core.attrs import find_xinv_coords,xinv_tp,xinv_st

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
@pytest.mark.parametrize("auxdim",["allaux","aux_0"])
def test_addneqs_same(neqbase,auxdim):
    """
    Test the adding of 2 normal equation systems with the same overlap of parameters 
    """
    if auxdim != "allaux":
        neqbase=neqbase.sel(naux=auxdim)

    neqbase2=neqbase.copy(deep=True)
    #add the two normal equation systems (both are the same)
    neqcomb_same=neqbase.xi.add(neqbase2)
   
    unk_xinvcoords=find_xinv_coords(neqbase,include=[xinv_tp.unk_co],state=xinv_st.linked)
    #Should be one only
    unk_name=next(iter(unk_xinvcoords.keys()))


    #note combined normal equation system may have a different sorting
    idx1=find_ilocs(neqcomb_same,unk_name,neqbase.coords[unk_name].data)
    
    #parameters should stay the same as there are no implicitly reduced parameters
    assert np.allclose(neqcomb_same.npara,neqbase.npara)
    #number of observations should be doubled
    assert np.allclose(neqcomb_same.nobs,2*neqbase.nobs)
    
    
    #Normal matrix should be twice the base input
    assert np.allclose(neqcomb_same.N[idx1,idx1],2*neqbase.N)

    #Right hand side should be twice the base input
    assert np.allclose(neqcomb_same.rhs[{unk_name:idx1}],2*neqbase.rhs)

    #LtpL should be twice the base input
    assert np.allclose(neqcomb_same.ltpl,2*neqbase.ltpl)
    
    #Sigma should be the same as the first base input
    assert np.allclose(neqcomb_same.sigma0,neqbase.sigma0)


@pytest.mark.parametrize("neqbase",["stacked"],indirect=True)
def test_addneq_compl(neqbase):
    """ 
    test adding two normal equation systems with a partial overlap of parameters
    """    
    neqbase2=neqbase.copy(deep=True)
    #modify the second normal equation system so that it has a different set of parameters but with an overlap
    neqbase2=neqbase2.xi.rename_groups(dict(poly='poly2'))


    #add the two normal equation systems (both are the same)
    neqcomb=neqbase.xi.add(neqbase2)
   
    #LtpL should be the sum of the base inputs
    assert np.allclose(neqcomb.ltpl,neqbase.ltpl+neqbase2.ltpl)
    
    #Sigma should be the same as the first base input
    assert np.allclose(neqcomb.sigma0,neqbase.sigma0)

    #number of parameters should be 10
    assert np.allclose([10]*neqcomb.sizes['naux'],neqcomb.npara)

    assert np.allclose(neqcomb.nobs,neqbase.nobs+neqbase2.nobs)
    
    #system 1 subset test
    neqc_poly=neqcomb.xi.get_group('poly')
    ixpoly=find_ilocs(neqbase,'poly',neqc_poly.poly.data)
    #polynomial coefficient entries should mathc the original ones of the first system
    assert np.allclose(neqbase.N[ixpoly,ixpoly],neqc_poly.N)
    assert np.allclose(neqbase.rhs[{'xinv_unk':ixpoly}],neqc_poly.rhs)

    #system 2 subset test
    neqc_poly=neqcomb.xi.get_group('poly2')
    ixpoly=find_ilocs(neqbase2,'poly2',neqc_poly.poly.data)
    #polynomial coefficient entries should mathc the original ones of the first system
    assert np.allclose(neqbase2.N[ixpoly,ixpoly],neqc_poly.N)
    assert np.allclose(neqbase2.rhs[{'xinv_unk':ixpoly}],neqc_poly.rhs)

    
    #check overlapping parameters
    neqc_seas=neqcomb.xi.get_group('harmonics_seasonal')
    neq1_seas=neqbase.xi.get_group('harmonics_seasonal')
    neq2_seas=neqbase2.xi.get_group('harmonics_seasonal')

    ixseas1=find_ilocs(neq1_seas,'harmonics_seasonal',neqc_seas.harmonics_seasonal.data)
    ixseas2=find_ilocs(neq2_seas,'harmonics_seasonal',neqc_seas.harmonics_seasonal.data)
    #Overlap should be the sum of the 2 systems
    assert np.allclose(neqc_seas.N,neq1_seas.N[ixseas1,ixseas1]+neq2_seas.N[ixseas2,ixseas2])
    
    assert np.allclose(neqc_seas.rhs,neq1_seas.rhs[{'harmonics_seasonal':ixseas1}]+neq2_seas.rhs[{'harmonics_seasonal':ixseas2}])




@pytest.mark.parametrize("neqbase",["simple"],indirect=True)
def test_addneqs_sigma(neqbase):
    """
    Test the adding of 2 normal equation systems with different apriori sigma0 
    """
    neqbase2=neqbase.copy(deep=True)


    # test for impossible sigma scaling (should fail)
    neqbase2['sigma0'][2]=0.1 #change only one value
    try:
        neqcomb=neqbase.xi.add(neqbase2)
        assert False, "Adding function should have raised an exception"
    except ValueError as exp:
        #ok, proceed with testing
        pass

    #modify sigma's of the second syst
    sigma_factor=2
    neqbase2['sigma0'][:]=sigma_factor*neqbase.sigma0
    
    pscale=1+1/(sigma_factor**2)

    #add the two normal equation systems (both are the same)
    neqcomb=neqbase.xi.add(neqbase2)
   
    unk_xinvcoords=find_xinv_coords(neqbase,include=[xinv_tp.unk_co],state=xinv_st.linked)
    # #Should be one only
    unk_name=next(iter(unk_xinvcoords.keys()))
    

    # #note combined normal equation system may have a different sorting
    idx1=find_ilocs(neqcomb,unk_name,neqbase.coords[unk_name].data)
    
    # #parameters should stay the same as there are no implicitly reduced parameters
    assert np.allclose(neqcomb.npara,neqbase.npara)
    # #number of observations should be doubled
    assert np.allclose(neqcomb.nobs,2*neqbase.nobs)
    
    
    # #Normal matrix should be pscale times the base input
    assert np.allclose(neqcomb.N[idx1,idx1],pscale*neqbase.N)

    # #Right hand side should be pscale times the base input
    assert np.allclose(neqcomb.rhs[{unk_name:idx1}],pscale*neqbase.rhs)

    # #LtpL should be pscale the base input
    assert np.allclose(neqcomb.ltpl,pscale*neqbase.ltpl)
    
    # #Sigma should be the same as the first base input
    assert np.allclose(neqcomb.sigma0,neqbase.sigma0)

