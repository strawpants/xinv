# This file is part of the xinv software which is licensed
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

# distutils: language = c++
# cython: profile=False

import numpy as np
cimport numpy as np
import xarray as xr
import os
from libc.stdlib cimport strtol,strtod
from xinv.core.logging import xinvlogger
from xinv.core.attrs import N_attrs, get_xunk_size_coname
import cython

#A bit of a hack to directly allow access to the data of a Python unicode str  object
cdef extern from *:
    void *PyUnicode_DATA(object unicode)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def read_symmat_fast(fileobj,dsout,blockname):
    """
        Reads a triangular matrix from a SINEX block and returns a symmetric version (Cython version)
        This version is optimized and uses the following tricks
        1. Direct access through a character pointer to the data of a unicode object (line of the files)
        2. Use of Cython memoryviews to avoid boiunds checking of the numpy arrays
        3. Use of Cython cdef variables to avoid type checking and optimize loop
        4. Use of stdlib strtol, and strod to parse the lines
    Parameters
    ----------
    fileobj : 
        io buffer to read lines from
        
    dsout : xarray.Dataset
        xarray.Dataset to augment the matrix data to
        
        
    blockname : str
        name of the SINEX block. should be one of:
        SOLUTION/NORMAL_EQUATION_MATRIX U
        SOLUTION/NORMAL_EQUATION_MATRIX L
        

    Returns
    -------
    an updated xarray.Dataset holding the new matrix in a new variable 'N'

    """
    if not blockname.startswith('SOLUTION/NORMAL_EQUATION_MATRIX'):
        raise RuntimeError(f"Wrong block {blockname}?")
    cdef int nest
    nest,xunk_co=get_xunk_size_coname(dsout)
    xunkdim=dsout[xunk_co].dims[0]
    mat=np.zeros([nest,nest],order='C')
    
    cdef double [:,:] cmat =mat
    cdef int irow
    cdef int icol
    cdef int ndat
    # cdef const unsigned char[:] ucline
    cdef char* cline
    cdef str line
    for line in fileobj:
        #get the char pointer to the buffer in the unicode object
        cline= <char*> PyUnicode_DATA(line)
        # cline = <const char*> &line[0]
        #cline = line[0]
        if cline[0] == '-':
            #end of block encountered
            break
        elif cline[0] == '*':
            #comment
            continue

        # sscanf(cline, " %5li %5li", &irow, &icol)
        irow=strtol(cline,&cline,10)

        icol=strtol(cline,&cline,10)
        irow-=1 #note zero indexing
        icol-=1

        ndat=nest-icol if icol>nest-3 else 3
        
        if ndat == 1:
            # #read one element
            cmat[irow,icol]=strtod(cline,&cline)
        elif ndat == 2:
            # #read 2 elements
            cmat[irow,icol]=strtod(cline,&cline)
            cmat[irow,icol+1]=strtod(cline,&cline)
            
        else:
            # #read 3 elements
            cmat[irow,icol]=strtod(cline,&cline)
            cmat[irow,icol+1]=strtod(cline,&cline)
            cmat[irow,icol+2]=strtod(cline,&cline)
    
    #mirror the upper triangle in the lower part
    mat=np.triu(mat,k=1).T+mat

    # if "nm_" not in dsout.indexes:
        # #add the transposed index
        # mi_=SHindexBase.mi_toggle(dsout.indexes['nm'])
        # dsout=dsout.sh.set_nmindex(mi_,'_')

    dsout['N']=([xunkdim,f"{xunkdim}_"],mat)
    dsout.N.attrs.update(N_attrs())
    return dsout

