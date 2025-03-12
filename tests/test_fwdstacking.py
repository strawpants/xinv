#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.polynomial import Polynomial
from xinv.fwd.fwdstack import FwdStackOp
from xinv.core.exceptions import XinvIllposedError

@pytest.fixture(params=['C','F'])
def noisystacked(request):
    """
    Generate a noisy dataset from a set of polynomial functions and annual and semi-annual signals
    """

    npoly=3
    noise_std=0.01
    naux=10 #number of auxiliary datasets
    x_axis=np.arange(-2,3,0.05)
    order=request.param[0]
    polyobs=np.zeros([naux,len(x_axis)],order=order)
    polytrue=np.zeros([naux,npoly+1])
    for i in range(naux):
        polytrue[i,:]=[j*i for j in range(npoly+1)]
        polyobs[i,:]=np.polyval(polytrue[i,::-1],x_axis)
    
    #add some normal noise
    polyobs+= np.random.normal(0,noise_std,polyobs.shape)
    
    # create a naming of the auxdims
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dspoly=xr.Dataset({"polyobs":xr.DataArray(polyobs,dims=("naux","x")),"polytrue":xr.DataArray(polytrue.T,dims=("poly","naux"))},coords={"x":x_axis,"naux":auxcoord,"poly":np.arange(npoly+1)},attrs=dict(noise_std=noise_std))

    return dspoly


def test_stacked_illposed(noisystacked):
    """
    Test a stacked forward operator, consisting of multiple stacked forward operators, building of a normal equation system, and solving step. This setup shoudl results in an illposed system and should be captured as such
    Parameters
    ----------
    noisystacked : xr.Dataset containing the noisy polynomial observations and the true polynomial coefficients

    """

    #initialize the first polynomial forward operator
    npoly=noisystacked.sizes['poly']-1
    polyfwd=Polynomial(n=npoly,poly_x='x',cache=True)
    
    #initialize the second polynomial forward operator (n=0, so illposed because its also contained within polyfwd)
    polyfwd2=Polynomial(n=0,poly_x='x',unknown_dim="poly2")
    #initialize the stacked forward operator
    fwdstck=FwdStackOp(polyfwd)
    fwdstck.append(polyfwd2)

    #Append Annual and Seminannual fwd operators
    # fwdstck.append(...)

    # #build the normal equation system
    std_noise=0.5
    dsneq=noisystacked.polyobs.xi.build_normal(fwdstck,ecov=std_noise*std_noise) 
    try:
        dssol=dsneq.xi.solve()
        #we're not supposed to end up here
        assert False
    except XinvIllposedError as e:
        assert True
        
def test_stacked(noisystacked):
    """
    Test a stacked forward operator, consisting of multiple stacked forward operators, building of a normal equation system, and solving step. 
    Parameters
    ----------
    noisystacked : xr.Dataset containing the noisy polynomial observations and the true polynomial coefficients and harmonic coefficients to test against

    """

    #initialize the first polynomial forward operator
    npoly=noisystacked.sizes['poly']-1
    polyfwd=Polynomial(n=npoly,poly_x='x',cache=True,x0=0)
    
    #initialize the stacked forward operator
    fwdstck=FwdStackOp(polyfwd)

    #Append Annual and Seminannual fwd operators
    # fwdstck.append(...)

    # #build the normal equation system
    std_noise=0.5
    dsneq=noisystacked.polyobs.xi.build_normal(fwdstck,ecov=std_noise*std_noise) 
    dssol=dsneq.xi.solve()

    #extract the groups of the solution and compare to the true values
    dsolpoly=dssol.xi.get_group('poly')
    prenoise=noisystacked.attrs['noise_std']
    tol=3*np.sqrt(np.diag(dsolpoly.COV))*prenoise
    tol=tol.max()
    assert np.allclose(dsolpoly.solution,noisystacked.polytrue,atol=tol)        
