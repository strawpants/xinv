## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from scipy.linalg import cholesky
from xinv.core.grouping import build_group_coord,find_group_coords


class CovarianceMat(CovarianceBase):
    def __init__(self,N_or_Cov="Normal or Covariance matrix",error_cov=True):


        if error_cov is True:
            cov=N_or_Cov
            N=xr.apply_ufunc(np.linalg.inv,cov)
        else:
            N=N_or_Cov
            cov=xr.apply_ufunc(np.linalg.inv,N)

        super().__init__(cov=cov,N=N)

        # get the covariance matrix for each block after splitting the transformed N into blocks -block wise covariance matrices
        #N = BTATPAB = BTNB

       # check if N already has group id
       grp_id_co, grp_seq_co = find_group_coords(N)

       if grp_id_co is None and grp_seq_co is None:
           
           unkdim=N.dims[0]

           grp_co=build_group_coord(N,dim=unkdim,group_id_name=group_id_co.name,group_seq_name=group_seq_co.name)
            
           
        self._group_id_dim=group_id_dim
        self._group_seq_dim=group_seq_dim

        

        dsout=xr.apply_ufunc(cholesky,Ngroup) ######### incomplete

        self._N=dsout

    def decorrelate(self,damat):       

        group_id_co,group_seq_co=find_group_coords(damat)
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

    




        