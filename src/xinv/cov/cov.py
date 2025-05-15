## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from scipy.linalg import cholesky

class CovarianceBase:
    def __init__(self):
        self._N=[]
        
    def decorrelate(self,damat):
        pass

class CovarianceMat(CovarianceBase):
    def __init__(self,N_or_Cov,error_cov=True):
        super().__init__()

        if error_cov is True:
            N=xr.apply_ufunc(np.linalg.inv,N_or_Cov)
        else:
            N=N_or_Cov

        self._N=xr.apply_ufunc(cholesky,N,input_core_dims=[["nm","nm_"]],output_core_dims=[["nm","nm_"]],kwargs={"lower":True})
        

    def decorrelate(self,damat):
        dsout=xr.dot(self._N,damat)

        return dsout

class DiagonalCovarianceMat(CovarianceBase):
    def __init__(self,diag_std):
        super().__init__()
        

        self._diag_std=xr.apply_ufunc(np.square,diag_std)  ## sigma** 2

    def decorrelate(self,damat):
        dsout=damat/self._diag_std  # A * sigma** -2 (diagonal N)
        return dsout
        
    