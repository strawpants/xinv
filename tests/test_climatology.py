#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.polynomial import Polynomial
from xinv.fwd.climatology import Climatology
from xinv.fwd.fwdstack import FwdStackOp
from xinv.neq.regularize import getMeanConstraint
from xinv.core.exceptions import XinvIllposedError
import os




# neqfile1=os.path.join(os.path.dirname(__file__),f'testdata/neqpoly.nc')
# neqfile_illposed=os.path.join(os.path.dirname(__file__),f'testdata/neqpoly_illposed.nc')

# #note apply a seed to garantee reproducibility (otherwise tests may fail in  statistical sense)
rg=np.random.default_rng(12789)

@pytest.fixture(params=['C','F'])
def noisystacked(request):
    """
    Generate a noisy dataset from a set of polynomial functions and a monthly climatology spread
    """

    npoly=1
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
    
    nmonth=12

    baseclim=[1,2,4,4,5,6,7,8,3,2,1,0]
    climatologytrue=np.zeros([naux,nmonth])
    
    t_months=t_axis.astype('datetime64[M]')-t_axis.astype('datetime64[Y]')
    #add monthly climatology
    for i in range(naux):
        #set climatology
        for j in range(nmonth):
            #let the max monthly value coincide with
            maxmonth=naux%nmonth
            max_ampl=naux
            #different aux contributors have a shifted and scaled behavior
            baseclimrolled=(i+1)*np.roll(baseclim,i)
            climatologytrue[i,j]=baseclimrolled[j]
    
        obs[i,:]+=climatologytrue[i,t_months.astype(int)]

    #add some normal noise
    obs+= rg.normal(0,noise_std,obs.shape)
    
    # create a naming of the auxdims
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dsobs=xr.Dataset({"obs":xr.DataArray(obs,dims=("naux","time")),"polytrue":xr.DataArray(polytrue.T,dims=("poly","naux")),"climtrue":xr.DataArray(climatologytrue.T,dims=("month","naux"))},coords={"time":t_axis,"naux":auxcoord},attrs=dict(noise_std=noise_std,t0=t0,delta_t=delta_t,npoly=npoly))

    return dsobs


def test_climatology(noisystacked):
    """
    Test a stacked forward operator, consisting of multiple stacked forward operators, building of a normal equation system, and solving step. This setup should results in an illposed system and should be captured as such
    Parameters
    ----------
    noisystacked : xr.Dataset containing the noisy polynomial observations and the true polynomial coefficients

    """
    
    #initialize the first polynomial forward operator
    npoly=noisystacked.attrs['npoly']
    t0=noisystacked.attrs['t0']
    delta_t=noisystacked.attrs['delta_t']
    polyfwd=Polynomial(n=npoly,poly_x='time',x0=t0,delta_x=delta_t,cache=True)
   
    #Initialize climatology forward operator
    climfwd=Climatology()

    #initialize the stacked forward operator
    fwdstck=FwdStackOp(polyfwd)
    fwdstck.append(climfwd)

    # # #build the normal equation system
    std_noise=0.5
    dsneq=noisystacked.obs.xi.build_normal(fwdstck,ecov=std_noise*std_noise) 
    try:
        dssol=dsneq.xi.solve()
        #we're not supposed to end up here because the polynomial mean and the climatalogy have a rank defect
        # if we do -> failed test        
        assert False
    except XinvIllposedError as e:
        # works as expected continue with fixing the degree 0 polynomial component
        pass
    
    dsneq=dsneq.xi.fix([('poly',0)])
    dssol=dsneq.xi.solve()
    # #extract the groups of the solution and compare to the true values
    dsolpoly=dssol.xi.get_group('poly')
    prenoise=noisystacked.attrs['noise_std']
    tol=3*np.sqrt(np.diag(dsolpoly.COV))*prenoise
    tol=tol.max()
    
    assert np.allclose(dsolpoly.solution,noisystacked.polytrue[1,:],atol=tol)

    dsolclim=dssol.xi.get_group('month')
    tol=3*np.sqrt(np.diag(dsolclim.COV))*prenoise
    tol=tol.max()
    assert np.allclose(dsolclim.solution,noisystacked.climtrue,atol=tol)


def test_climatology_mean0(noisystacked):
    """
    Test a stacked forward operator, consisting of multiple stacked forward operators, building of a normal equation system, and solving step. This setup should results in an illposed system and should be captured as such
    Parameters
    ----------
    noisystacked : xr.Dataset containing the noisy polynomial observations and the true polynomial coefficients

    """
    
    #initialize the first polynomial forward operator
    npoly=noisystacked.attrs['npoly']
    t0=noisystacked.attrs['t0']
    delta_t=noisystacked.attrs['delta_t']
    polyfwd=Polynomial(n=npoly,poly_x='time',x0=t0,delta_x=delta_t,cache=True)
   
    #Initialize climatology forward operator
    climfwd=Climatology()

    #initialize the stacked forward operator
    fwdstck=FwdStackOp(polyfwd)
    fwdstck.append(climfwd)

    # # #build the normal equation system
    std_noise=0.5
    dsneq=noisystacked.obs.xi.build_normal(fwdstck,ecov=std_noise*std_noise) 
    try:
        dssol=dsneq.xi.solve()
        #we're not supposed to end up here because the polynomial mean and the climatalogy have a rank defect
        # if we do -> failed test        
        assert False
    except XinvIllposedError as e:
        # works as expected continue with fixing the degree 0 polynomial component
        pass
   
    #add a mean constraint
    constrcoord=dsneq.xinv_unk.loc[dict(xinv_unk=('month',[0,1,2,3,4,5,6,7,8,9,10,11]))]
    dsmean=getMeanConstraint(constrcoord)
    dsbasereg=dsneq.xi.reg(dsmean,alpha=1e3)
    dssol=dsbasereg.xi.solve()
    # #extract the groups of the solution and compare to the true values
    dsolpoly=dssol.xi.get_group('poly')
    prenoise=noisystacked.attrs['noise_std']
    tol=3*np.sqrt(np.diag(dsolpoly.COV))*prenoise
    tol=tol.max()
   
    #check whether the mean is resolved
    assert np.allclose(dssol.solution[0,:],noisystacked.polytrue[0,:]+noisystacked.climtrue.mean('month'),atol=tol)
    
    assert np.allclose(dssol.solution[1,:],noisystacked.polytrue[1,:],atol=tol)

    dsolclim=dssol.xi.get_group('month')
    tol=3*np.sqrt(np.diag(dsolclim.COV))*prenoise
    tol=tol.max()
    assert np.allclose(noisystacked.climtrue.T,noisystacked.climtrue.mean('month')+dssol.solution[2:,:],atol=tol)

    #check whether the estimated mean is small
    assert np.abs(dsolclim.solution.mean('month')).max() < 1e-10


