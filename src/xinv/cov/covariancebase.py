## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import xarray as xr

class CovarianceBase:
    def __init__(self,N=None,cache=False,decorrname="decorrelated"):
      #  self._cov=None
        self._N=N
        self._cache_decorrelate=cache
        self._decorr=None
        self._decorrname=decorrname

    def decorrelate(self,damat=None):
        """Decorrelate the damat using cholesky of the normal matrix"""
        
        if self._N is None and damat is None:
            raise ValueError("Requesting the decorrelated damat requires a valid covariance matrix")
        elif damat is None:
            return self._decorr



        decorr=self._decorrelate_impl(damat)
        if type(decorr) == xr.DataArray:
            decorr.name=self._decorrname
            decorr=decorr.to_dataset()

        if self._cache_decorrelate:
            self._decorr=decorr
            
        return decorr
