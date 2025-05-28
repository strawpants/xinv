## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import pytest
import xarray as xr
import numpy as np
from xinv.cov.cov import CovarianceMat,DiagonalCovarianceMat
from xinv.fwd.polynomial import Polynomial
import os
#from xinv import *

rg=np.random.default_rng()

@pytest.fixture(params=['C','F'])
def noisypoly(request):
    """ 
    Generate a noisy dataset from a set of polynomial functions
    """
    npoly=3
    noise_std=0.01
    x0=0
    delta_x=2
    naux=10
    x_axis=np.arange(-2,3,0.05)
    x_axis_rel=(x_axis-x0)/delta_x
    order=request.param[0]
    polyobs=np.zeros([naux,len(x_axis)],order=order)
    polytrue=np.zeros([naux,npoly+1])
    damat=np.stack([x_axis_rel**i for i in range(npoly+1)],axis=1)

    for i in range(naux):
        polytrue[i,:]=[j*i for j in range(npoly+1)]
        polyobs[i,:]=np.polyval(polytrue[i,::-1],x_axis_rel)

    polyobs+=rg.normal(0,noise_std,polyobs.shape)

    # create a naming of the auxdims
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dspoly=xr.Dataset({"polyobs":xr.DataArray(polyobs,dims=("naux","nm")),"polytrue":xr.DataArray(polytrue,dims=("naux","poly")),"damat":xr.DataArray(damat,dims=("nm","poly"))},coords={"naux":auxcoord,"nm":x_axis,"poly":np.arange(npoly+1)},attrs=dict(noise_std=noise_std,x0=x0,delta_x=delta_x))


    return dspoly


def poly_covariance(noisypoly):
    """
    Build a diagonal covariance matrix and standard deviation vector
    """
    std=noisypoly.damat.std(dim="poly")
    var=std**2
    
    std_da=xr.DataArray(std.data,dims="nm",coords={"nm":noisypoly.nm})
    cov_da=xr.DataArray(np.diag(var.data),dims=("nm","nm_"),coords={"nm":noisypoly.nm.data,"nm_":noisypoly.nm.data})

    return cov_da,std_da




def test_poly_decorrelation(noisypoly):
    """
    Compare decorrelation using a full diagonal covariance matrix vs. diagonal std-based decorrelation
    """
    cov_da,std_da=poly_covariance(noisypoly)

    cov_full=CovarianceMat(N_or_Cov=cov_da,error_cov=True)
    decorr_full=cov_full.decorrelate(noisypoly.damat).decorrelated
    
    cov_diag=DiagonalCovarianceMat(diag_std=std_da)
    decorr_diag=cov_diag.decorrelate(noisypoly.damat).decorrelated


    assert np.allclose(decorr_full.data,decorr_diag,atol=1e-15)




    
