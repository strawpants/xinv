# This file is part of the shxarray software which is licensed
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl


from gzip import GzipFile
import os
from io import TextIOWrapper,BytesIO

try:
    from rapidgzip import RapidgzipFile
except:
    #failed to load (ok)
    RapidgzipFile=None

def gzip_open_r(filename,textmode=False):
    """
        GZip file reading wrapper leveraging parallel decompression speed when rapidgzip is installed on the system
    Parameters
    ----------
    filename : str
        Filename of the gzip archive to open
        
    textmode : 
        Whether to open in textmode (allows iterating over lines)
        
    """
    
    if filename.endswith('.Z'):
        
        try:
            import unlzw3
        except ImportError:
            raise ImportError("unlzw3 is not installed, please install it to use the .Z file format")
        #use uncompress
        with open(filename, 'rb') as f:
            #rtead entire file
            gzfid = BytesIO(unlzw3.unlzw(f.read()))


    elif RapidgzipFile: 
        gzfid=RapidgzipFile(filename,parallelization=os.cpu_count())

    else:
        gzfid=GzipFile(filename,'rb')


    if textmode:
        return TextIOWrapper(gzfid)
    else:
        return gzfid



