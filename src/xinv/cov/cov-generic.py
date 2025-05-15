## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr
from scipy.linalg import cholesky
from xinv.core.grouping import expand_as_group

import numpy as np
import xarray as xr
from scipy.linalg import cholesky
from xinv.core.grouping import expand_as_group

class CovarianceBase:
    def __init__(self):
        self._cov=[]

    def decorrelate(self, damat):
        pass
    

class CovarianceMat(CovarianceBase):
    def __init__(self,N_or_Cov,error_cov=True,group_id_dim="xinv_grp_id",group_seq_dim="xinv_grp_seq"):
        super().__init__()
        self._group_id_dim=group_id_dim
        self._group_seq_dim=group_seq_dim


        if error_cov is True:
            N=xr.apply_ufunc(np.linalg.inv,N_or_Cov)
        else:
            N=N_or_Cov

        row_dim,col_dim=N.dims

        Nout=N.to_dataset(name="Nmat")
        Ngroup=expand_as_group(Nout,group_dim=row_dim,group_id_dim=group_id_dim,group_seq_dim=group_seq_dim)



        dsout=xr.apply_ufunc(cholesky,Ngroup[0]) ######### incomplete

        self._N=dsout

    def decorrelate(self,damat): 
        pass

    




        