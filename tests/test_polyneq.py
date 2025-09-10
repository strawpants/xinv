#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.polynomial import Polynomial
import os

from fixtures import noisypoly

neqpolyf=os.path.join(os.path.dirname(__file__),f'testdata/neqpoly_simple.nc')



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
    if not os.path.exists(neqpolyf):
        #save the normal equation system to a file (for use in other tests)
        dsneq.to_netcdf(neqpolyf)

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

