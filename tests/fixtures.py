## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl


# fixtures shared among test cases

import pytest
import numpy as np
import xarray as xr

# we fix the seed for the noise generator below to make it noisy but reproducible
rg=np.random.default_rng(12789)

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
    polyobs+= rg.normal(0,noise_std,polyobs.shape)
    
    # create a naming of the auxdims
    auxcoord=[f"aux_{i}" for i in range(naux)]
    dspoly=xr.Dataset({"polyobs":xr.DataArray(polyobs,dims=("naux","x")),"polytrue":xr.DataArray(polytrue.T,dims=("poly","naux"))},coords={"x":x_axis,"naux":auxcoord,"poly":np.arange(npoly+1)},attrs=dict(noise_std=noise_std,x0=x0,delta_x=delta_x))

    return dspoly


