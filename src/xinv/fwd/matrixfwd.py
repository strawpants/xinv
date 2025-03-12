
from xinv.fwd import FwdOpbase

class MatrixfwdOp(fwdOpbase):
    
    def __init__(self,dajac):
        
        super().__init__()
        self._dajac= dajac
        
        
    def _jacobian_impl(self,dain):      
        
        return self._dajac
            