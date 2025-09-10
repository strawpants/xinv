## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl

import xarray as xr 
from xinv.fwd import FwdOpbase
from xinv.core.attrs import xunk_coords_attrs
class MatrixfwdOp(FwdOpbase):
    def __init__(self,dajac:xr.DataArray):
        obs_dim=dajac.dims[0]
        unknown_dim=dajac.dims[1]
        super().__init__(obs_dim=obs_dim,unknown_dim=unknown_dim,cache=True)
        #sets the cached Jacobian matrix, so _jacobian_impl never gets called 
        self._jacob=dajac.to_dataset(name=self._jacobname)
        if self._unkdim in dajac.coords:
            self._jacob[self._unkdim].attrs.update(xunk_coords_attrs(state='linked'))








            
