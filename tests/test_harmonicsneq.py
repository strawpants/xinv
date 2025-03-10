#!/usr/bin/env python3
## Permissions: See the xinv license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.harmonics import Harmonics


@pytest.fixture(params=['C','F'])
def noisyharmonics(request):
    """ Generate a noisy dataset from a set of sine and cosine functions """
    
    nannual =1
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
 
        if i<6:
            trend = -(i+1)*trend_slope*x_axis
            hsobs[i,:]= trend + sinn + coss
            hstrue[i,0]= np.sqrt((3*scale)**2 + (-5*scale)**2)
            
        if i>=6:
            trend = (i+1)*trend_slope*x_axis
            hsobs[i,:]= trend + sinn + coss  
            hstrue[i,0]= np.sqrt((3*scale)**2 + (-5*scale)**2)
   
                
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

    nharmonics = 1 
    harmonic_fwd = Harmonics(n=nharmonics, semi_annual=False, annual_x='x', cache=True)

    std_noise = 0.01
    dsneq = noisyharmonics.hsobs.xi.build_normal(harmonic_fwd, ecov=std_noise * std_noise)  

    dssol = dsneq.xi.solve()

    prenoise = noisyharmonics.attrs['noise_std']

    assert np.allclose(dssol.sigma0, prenoise, rtol=0.25)

    tol = 3 * np.sqrt(np.diag(dssol.COV)) * prenoise
    tol = tol.max()

    assert np.allclose(dssol.solution, noisyharmonics.hstrue, atol=tol)              

    fwdobs = harmonic_fwd(dssol.solution)

    np.allclose(noisyharmonics.hsobs, fwdobs.T, atol=6 * prenoise)
