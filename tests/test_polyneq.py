#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.polynomial import Polynomial

@pytest.fixture(params=['C','F'])
def noisypoly(request):
    """
    Generate a noisy dataset from a set of polynomial functions
    """

    npoly=3
    noise_std=0.01
    x0=0
    delta_x=2
    naux=10 #number of auxiliary datasets
    x_axis=np.arange(-2,3,0.05)
    x_axis_rel=(x_axis-x0)/delta_x
    order=request.param[0]
    polyobs=np.zeros([naux,len(x_axis)],order=order)
    polytrue=np.zeros([naux,npoly+1])
    for i in range(naux):
        polytrue[i,:]=[j*i for j in range(npoly+1)]
        polyobs[i,:]=np.polyval(polytrue[i,::-1],x_axis_rel)
    
    #add some normal noise
    polyobs+= np.random.normal(0,noise_std,polyobs.shape)
    
    # create a naming of the auxdims
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dspoly=xr.Dataset({"polyobs":xr.DataArray(polyobs,dims=("naux","x")),"polytrue":xr.DataArray(polytrue.T,dims=("poly","naux"))},coords={"x":x_axis,"naux":auxcoord,"poly":np.arange(npoly+1)},attrs=dict(noise_std=noise_std,x0=x0,delta_x=delta_x))

    return dspoly


def test_polyneqs(noisypoly):
    """
    Test the polynomial forward operator, building of a normal equation system, and solving step
    Parameters
    ----------
    noisypoly : xr.Dataset containing the noisy polynomial observations and the true polynomial coefficients

    """

    #initialize the forward operator
    x0=noisypoly.attrs['x0']
    deltax=noisypoly.attrs['delta_x']
    npoly=noisypoly.sizes['poly']-1
    polyfwd=Polynomial(n=npoly,poly_x='x',x0=x0,delta_x=deltax,cache=True)
    
    #build the normal equation system
    std_noise=0.5
    dsneq=noisypoly.polyobs.xi.build_normal(polyfwd,ecov=std_noise*std_noise) 

    dssol=dsneq.xi.solve()
    
    #1. check whether the posteriori sigma is decently estimated
    prenoise=noisypoly.attrs['noise_std']

    #3 sigma test
    assert np.allclose(dssol.sigma0,prenoise,rtol=0.25)

    tol=3*np.sqrt(np.diag(dssol.COV))*prenoise
    tol=tol.max()

    assert np.allclose(dssol.solution,noisypoly.polytrue,atol=tol)        

    #4. propagate the solution to the observations

    fwdobs=polyfwd(dssol.solution)
    # test for containment within 6 sigma 
    np.allclose(noisypoly.polyobs,fwdobs.T,atol=6*prenoise) 

