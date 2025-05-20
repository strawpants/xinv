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

    for i in range(naux):
        polytrue[i,:]=[j*i for j in range(npoly+1)]
        polyobs[i,:]=np.polyval(polytrue[i,::-1],x_axis_rel)

    polyobs+=rg.normal(0,noise_std,polyobs.shape)

    # create a naming of the auxdims
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dspoly=xr.Dataset({"polyobs":xr.DataArray(polyobs,dims=("naux_","x")),"polytrue":xr.DataArray(polytrue.T,dims=("poly","naux_"))},coords={"x":x_axis,"naux_":auxcoord,"poly":np.arange(npoly+1)},attrs=dict(noise_std=noise_std,x0=x0,delta_x=delta_x))


    return dspoly


def build_poly_covariance(noisypoly):
       """
       Build a diagonal covariance matrix and standard deviation vector for polyobs.
       """
       std=noisypoly.polyobs.std(dim="x")
       var=std**2
    
       std_da=xr.DataArray(std.data,dims="naux_",coords={"naux_":noisypoly.naux_})
       cov_da=xr.DataArray(np.diag(var.data),dims=("naux_","naux"),coords={"naux_":noisypoly.naux_.data,"naux":noisypoly.naux_.data})

       return cov_da,std_da




def test_poly_decorrelation_consistency(noisypoly):
     """
     Compare decorrelation using a full diagonal covariance matrix vs. diagonal std-based decorrelation
     on the polyobs data
     """
     cov_da,std_da=build_poly_covariance(noisypoly)
     damat=noisypoly.polyobs  
     cov_full=CovarianceMat(N_or_Cov=cov_da,error_cov=True)
    
     decorr_full=cov_full.decorrelate(damat).decorrelated
     cov_diag=DiagonalCovarianceMat(diag_std=std_da)
     decorr_diag=cov_diag.decorrelate(damat).decorrelated

     xr.testing.assert_equal(decorr_full,decorr_diag)




    