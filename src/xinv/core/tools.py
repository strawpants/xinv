## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np

def find_ilocs(dsneq,dim,elements):
    idxname=f"_{dim}_idx"
    if idxname not in dsneq.keys():
        dsneq[idxname]=(dim,np.arange(dsneq.sizes[dim]))
    return dsneq[idxname].loc[elements].data

