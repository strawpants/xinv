## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl



from xarray.backends import BackendEntrypoint
from xinv.io.sinex import read_sinex

class SINEXBackEntryPoint(BackendEntrypoint):
    url="https://github.com/strawpants/xinv"
    description = "Read normal equation systems/solutions in SINEX (2.02) format"
    def open_dataset(self,filename_or_obj,*,drop_variables=None):
        if drop_variables is not None and "N" in drop_variables:
            #Special case: it is much quicker to abandon reading when a matrix is present and not needed
            dsout=read_sinex(filename_or_obj,stopatmat=True)
            drop_variables
        elif drop_variables is not None:
            dsout=read_sinex(filename_or_obj)
            dsout=dsout.drop_vars(drop_variables)
        else:
            dsout=read_sinex(filename_or_obj)
        
        return dsout
    
    def guess_can_open(self,filename_or_obj):
        try:
            strrep=str(filename_or_obj).lower()
            # search for usual file naming of SINEX files
            if strrep.endswith('.snx') or strrep.endswith('.snx.gz') or strrep.endswith('.snx.z'):  
                #Found a file name which probably is a sinex file
                return True
        except AttributeError:
            return False
            
        return False

