#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.polynomial import Polynomial
from xinv.fwd.harmonics import SeasonalHarmonics
from xinv.fwd.fwdstack import FwdStackOp
from xinv.core.exceptions import XinvIllposedError

@pytest.fixture(params=['C','F'])
def noisystacked(request):
    """
    Generate a noisy dataset from a set of polynomial functions and annual and semi-annual signals
    """

    npoly=2
    noise_std=0.01
    naux=5 #number of auxiliary datasets
    dtype='datetime64[D]'
    t_axis=np.arange('2001-01-31', '2020-12-31', dtype=dtype)
    t0=np.datetime64("2010-01-01",'D') 
    delta_t=np.timedelta64(365,'D')+ np.timedelta64(6,'h')
    order=request.param[0]
    obs=np.zeros([naux,len(t_axis)],order=order)
    
    t_axis_rel=(t_axis-t0)/delta_t #relative time axis
    #add polynomial truth values
    polytrue=np.zeros([naux,npoly+1])
    for i in range(naux):
        polytrue[i,:]=[j*i for j in range(npoly+1)]
        obs[i,:]=np.polyval(polytrue[i,::-1],t_axis_rel)
    
    nharm=2
    seastrue=np.zeros([naux,nharm*2])
    omegas=[2*np.pi,4*np.pi]
    frac=0.25 #fraction of the amplitude
    #add seasonal harmonics
    for i in range(naux):
        #set cosine and sine amplitudes
        camp=naux
        samp=1-naux
        scamp=frac*samp
        ssamp=frac*samp
        seastrue[i,:]= [camp,samp,ssamp,scamp] 
        obs[i,:]+=camp*np.cos(omegas[0]*t_axis_rel)+samp*np.sin(omegas[0]*t_axis_rel)+scamp*np.cos(omegas[1]*t_axis_rel)+ssamp*np.sin(omegas[1]*t_axis_rel)

    #add some normal noise
    obs+= np.random.normal(0,noise_std,obs.shape)
    
    # create a naming of the auxdims
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dsobs=xr.Dataset({"obs":xr.DataArray(obs,dims=("naux","time")),"polytrue":xr.DataArray(polytrue.T,dims=("poly","naux")),"seastrue":xr.DataArray(seastrue.T,dims=("seas","naux"))},coords={"time":t_axis,"naux":auxcoord},attrs=dict(noise_std=noise_std,t0=t0,delta_t=delta_t,npoly=npoly))

    return dsobs


def test_illposed(noisystacked):
    """
    Test a stacked forward operator, consisting of multiple stacked forward operators, building of a normal equation system, and solving step. This setup shoudl results in an illposed system and should be captured as such
    Parameters
    ----------
    noisystacked : xr.Dataset containing the noisy polynomial observations and the true polynomial coefficients

    """
    #initialize the first polynomial forward operator
    npoly=noisystacked.attrs['npoly']
    t0=noisystacked.attrs['t0']
    delta_t=noisystacked.attrs['delta_t']
    polyfwd=Polynomial(n=npoly,poly_x='time',x0=t0,delta_x=delta_t,cache=True)
    
    #initialize the second polynomial forward operator (n=0, so illposed because its also contained within polyfwd)
    polyfwd2=Polynomial(n=0,poly_x='time',x0=t0,delta_x=delta_t,unknown_dim="poly2")
    #initialize the stacked forward operator
    fwdstck=FwdStackOp(polyfwd)
    fwdstck.append(polyfwd2)

    # #build the normal equation system
    std_noise=0.5
    dsneq=noisystacked.obs.xi.build_normal(fwdstck,ecov=std_noise*std_noise) 
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
    npoly=noisystacked.attrs['npoly']
    t0=noisystacked.attrs['t0']
    delta_t=noisystacked.attrs['delta_t']
    polyfwd=Polynomial(n=npoly,poly_x='time',cache=True,x0=t0,delta_x=delta_t)
    
    #initialize the stacked forward operator
    fwdstck=FwdStackOp(polyfwd)

    #Append Annual and Seminannual fwd operators
    seasfwd=SeasonalHarmonics(x0=t0,semi_annual=True)
    fwdstck.append(seasfwd)

    # #build the normal equation system
    std_noise=0.5
    dsneq=noisystacked.obs.xi.build_normal(fwdstck,ecov=std_noise*std_noise) 
    dssol=dsneq.xi.solve()

    #extract the groups of the solution and compare to the true values
    dsolpoly=dssol.xi.get_group('poly')
    prenoise=noisystacked.attrs['noise_std']
    tol=3*np.sqrt(np.diag(dsolpoly.COV))*prenoise
    tol=tol.max()
    assert np.allclose(dsolpoly.solution,noisystacked.polytrue,atol=tol)

    dsolseas=dssol.xi.get_group('harmonics_seasonal')
    tol=3*np.sqrt(np.diag(dsolseas.COV))*prenoise
    tol=tol.max()
    assert np.allclose(dsolseas.solution,noisystacked.seastrue,atol=tol)
