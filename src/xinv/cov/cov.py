## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from scipy.linalg import cholesky

class CovarianceMat(CovarianceBase):
    def __init__(self,N_or_Cov="Normal or Covariance matrix",error_cov=True):

        if error_cov is True:
            cov=N_or_Cov
            N=xr.apply_ufunc(np.linalg.inv,cov)
            #,vectorize=True
        else:
            N=N_or_Cov
            cov=xr.apply_ufunc(np.linalg.inv,N)

        super().__init__(cov=cov,N=N)

        self._Ncholesky=xr.apply_ufunc(cholesky,N,input_core_dims=[["nm","nm_"]],output_core_dims=[["nm","nm_"]],kwargs={"lower":True})

    def decorrelate(self,damat):        
        dsout=xr.dot(self._Ncholesky,damat)

        return dsout

class DiagonalCovarianceMat(CovarianceBase):
    def __init__(self,diag_std):

        cov=xr.apply_ufunc(np.square,diag_std) 
        N=1/cov                
        super().__init__(cov=cov,N=N)        
        self._var=cov
    
    def decorrelate(self,damat):
        
        dsout=damat/self._var 
        return dsout
        
    