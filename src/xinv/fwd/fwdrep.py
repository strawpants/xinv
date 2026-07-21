## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr
import numpy as np
from xinv.fwd import FwdOpbase
from scipy.sparse import block_diag
from sparse import as_coo # needed to embed sparse arrays in xarray
from xinv.core.logging import xinvlogger as logger
from xinv.core.attrs import find_xinv_coords,xinv_st
import pandas as pd

class FwdRepOp(FwdOpbase):
    def __init__(self,fwdsrc=None,cache=False,rep_dim=None,unknown_dim="xrep_unk",**bindargs):
        """Setup a forward operator which repeats the source forward operator along the give dimension"""
        super().__init__(cache=cache,unknown_dim=unknown_dim,obs_dim=rep_dim,**bindargs)
        self._fwdsrc=fwdsrc
        self._repdim=rep_dim
    def _jacobian_impl(self,**kwargs):
        """Creates the Jacobian of the forward operator"""
        

        #fifure out the coordinate to repeat along
        if self._repdim in kwargs:
            repcoords=kwargs[self._repdim]
            if type(repcoords) != xr.DataArray:
                repcoords=xr.DataArray(repcoords,dims=self._repdim)
        elif "daobs" in kwargs:
            repcoords=kwargs['daobs'].coords[self._repdim]

        jac_src=self._fwdsrc.jacobian(**kwargs)
        if jac_src.jacobian.shape[0] != 1:
            msg="FwdReOp cannot deal with source Jacobians with more than one row"
            logger.error(msg)
            raise ValueError(msg)

        src_unkco=jac_src[self._fwdsrc.unkdim]

        nreps=repcoords.sizes[self._repdim]
        #build sparse matrix (note we need to use sparse.COO format as it is the only one currently supported by xarray)
        spmat=as_coo(block_diag([jac_src.jacobian.data for i in range(nreps)],format='coo'))
        
        #create new index vector spanning the new unknown dims
        unkco_prod=pd.MultiIndex.from_product([repcoords[self._obsdim].values,src_unkco.values],names=[self._repdim+"_rep",self._fwdsrc._unkdim])
        
        coords_adopt={self._obsdim:repcoords,self._unkdim:unkco_prod}
        #also adopt unlinked xinv coordinates for bookkeeping purposes
        coords_adopt.update(find_xinv_coords(jac_src,state=xinv_st.unlinked))


        jacobian=xr.Dataset(dict(jacobian=([self._obsdim,self._unkdim],spmat)),coords=coords_adopt)
        return jacobian

