## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from scipy.linalg import cholesky

from xinv.cov import covariancebase
from xinv.cov.covariancebase import CovarianceBase

class CovarianceMat(CovarianceBase):
    def __init__(self,N_or_Cov="Normal or Covariance matrix",error_cov=True):

        if error_cov is True:
            cov=N_or_Cov
            N=xr.apply_ufunc(np.linalg.inv,cov)
        else:
            N=N_or_Cov
            cov=xr.apply_ufunc(np.linalg.inv,N)

        super().__init__(N=N)

        naux_,naux=N.dims

        
        self._Ncholesky=xr.apply_ufunc(cholesky,N,input_core_dims=[["naux_","naux"]],output_core_dims=[["naux_","naux_"]],kwargs={"lower":True})


    
    def _decorrelate_impl(self,damat):        
        
        decorrelated=xr.dot(self._Ncholesky,damat)

        return decorrelated

class DiagonalCovarianceMat(CovarianceBase):
    def __init__(self,diag_std):

        var=xr.apply_ufunc(np.square,diag_std) 
        N=1/var                
        super().__init__(N=N)        
        self._var=var
    
    def _decorrelate_impl(self,damat):
        
        decorrelated=damat/self._var 

        return decorrelated
        
    