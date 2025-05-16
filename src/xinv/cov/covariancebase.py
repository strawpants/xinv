## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

class CovarianceBase:
    def __init__(self,cov=None,N=None):
        self._cov=cov
        self._N=N

    def decorrelate(self,damat):
        raise NotImplementedError
