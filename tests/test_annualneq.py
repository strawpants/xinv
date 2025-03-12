#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.annual import Annual

@pytest.fixture(params=['C','F'])
def noisyharmonics(request):
    """ Generate a noisy dataset from a set of sine and cosine functions """
    
    nannual =2
    naux=10 # number of auxiliary datasets
    noise_std=0.01
    trend_slope= 0 
    order=request.param[0]
    
    #x_axis = np.arange(-10,10, 0.05)
    t0_date = np.datetime64('2012-01-01')
    T_yearly = np.timedelta64(365, 'D') + np.timedelta64(6, 'h')  
    random_dates = t0_date + np.arange(0, 365 * 22, 0.05).astype('timedelta64[D]')
    x_axis = (random_dates - t0_date) / T_yearly
    
    
    hstrue=np.zeros([naux,nannual])
    hsobs = np.zeros([naux,len(x_axis)],order=order)
    
    for i in range(naux):
        scale =  (i/naux)*np.pi
        omega = (2*np.pi)
        coss= (3*scale)*np.cos(omega*x_axis)
        sinn= (-5*scale)*np.sin(omega*x_axis)
        
        # phase = (i/naux)*np.pi
        # coss= 10*np.cos(2*x_axis+phase)
        # sinn= -5*np.sin(5*x_axis+phase)
        if i<6:
            trend = -(i+1)*trend_slope*x_axis
            hsobs[i,:]= trend + sinn + coss
            hstrue[i,0]= (3*scale)
            hstrue[i,1]= (-5*scale)
        if i>=6:
            trend = (i+1)*trend_slope*x_axis
            hsobs[i,:]= trend + sinn + coss  
            hstrue[i,0]= (3*scale)
            hstrue[i,1]= (-5*scale)
                
    hsobs+= np.random.normal(0,noise_std, hsobs.shape)
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dshs=xr.Dataset({"hsobs":xr.DataArray(hsobs,dims=("naux","x")),"hstrue":xr.DataArray(hstrue.T,dims=("nannual","naux"))},coords={"x":x_axis,"naux":auxcoord,"nannual":np.arange(nannual)},attrs=dict(noise_std=noise_std))

    return dshs


def test_noisyharmonics(noisyharmonics):
    """
    Test the annual forward operator, building of a normal equation system, and solving step
    Parameters
    ----------
    noisyharmonics : xr.Dataset containing the noisy harmonic observations and the true sine/cosine coefficients

    """

    #initialize the forward operator
    nharmonics = 2  
    harmonic_fwd = Annual(n=nharmonics,annual_x='x',cache=True)

    # Build the normal equation system
    std_noise = 0.01
    dsneq = noisyharmonics.hsobs.xi.build_normal(harmonic_fwd,ecov=std_noise*std_noise)  ### check the std values

    dssol = dsneq.xi.solve()

    # Check whether the posteriori sigma is correctly estimated
    prenoise = noisyharmonics.attrs['noise_std']

    # 3-sigma test for uncertainty validation
    assert np.allclose(dssol.sigma0,prenoise,rtol=0.25)

    # Compute error tolerance
    tol = 3 * np.sqrt(np.diag(dssol.COV))*prenoise
    tol = tol.max()

    assert np.allclose(dssol.solution,noisyharmonics.hstrue,atol=tol)              

    fwdobs = harmonic_fwd(dssol.solution)

    np.allclose(noisyharmonics.hsobs, fwdobs.T, atol=6 * prenoise)


