## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl, 2025 Kiana Karimi, s.karimi@utwente.nl

from xinv.fwd import FwdOpbase

class MatrixfwdOp(fwdOpbase):
    
    def __init__(self,dajac):
        
        super().__init__()
        self._dajac= dajac
        
        
    def _jacobian_impl(self,dain):      
        
        return self._dajac
            