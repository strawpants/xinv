## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import xarray as xr
from xinv.core.attrs import islower,Chol_attrs,xinv_st,cov_attrs
from xinv.core.logging import xinvlogger
import numpy as np
from xinv.core.exceptions import XinvIllposedError

from scipy.linalg import cholesky
from scipy.linalg.lapack import dpotri
from scipy.linalg.blas import dtrsm,dsyrk


from enum import Enum

class MemLayout(Enum):
    F_cont = 1
    C_cont = 0
    non_cont = -1

def memlayout(arr:np.ndarray):
    if arr.flags['F_CONTIGUOUS'] and arr.strides[0] == arr.dtype.itemsize:
        return MemLayout.F_cont
    elif arr.flags['C_CONTIGUOUS'] and arr.strides[1] == arr.dtype.itemsize:
        return MemLayout.C_cont
    else:
        return MemLayout.non_cont


def cholesky_inplace(N:xr.DataArray):
    """
        Inplace Cholesky decomposition of a symmetric positive definite matrix stored in an xarray DataArray. 
        The input matrix N is overwritten with its Cholesky factor.

    """


    if N.attrs['xinv_state'] not in [xinv_st.symU,xinv_st.symL]:
        raise RuntimeError(f"Don't know how to do a Cholesky decomposition on this xinv state :{N.attrs['xinv_state']}")
   
    lower=islower(N)
    restore=False

    n_layout=memlayout(N.data)

    #check for C or F ordering
    if n_layout == MemLayout.F_cont:
        #fine F ordered array with stride 1 of double values
        lowapparent=lower
        #pointer
        Ndat=N.data
    elif n_layout == MemLayout.C_cont:
        #switch lower/upper representation in the F-ordered matrix of doubl
        lowapparent=1-lower
        Ndat=N.data.T
    else:
        xinvlogger.warning("Normal matrix is not C or F contiguous, applying copy-restore")
        lowapparent=lower
        restore=True
        Ndat=N.data.copy(order='F')

    try:
        cholesky(Ndat,lower=lowapparent,overwrite_a=1)
        if restore:
            #copy data back
            N[()]=Ndat
    except np.linalg.LinAlgError as e:
        raise XinvIllposedError(str(e))
    N.attrs.update(Chol_attrs(lower))

    return N #same object (will be changed now)

def dpotri_inplace(N:xr.DataArray):
    lower=islower(N)

    restore=False

    n_layout=memlayout(N.data)
    #check for C or F ordering
    if n_layout == MemLayout.F_cont:
        #fine F ordered array with stride 1 of double values
        lowapparent=lower
        #pointer
        Ndat=N.data
    elif n_layout == MemLayout.C_cont:
        #switch lower/upper representation in the F-ordered matrix of doubl
        lowapparent=1-lower
        Ndat=N.data.T
    else:
        xinvlogger.warning("Normal matrix is not C or F contiguous, applying copy-restore")
        lowapparent=lower
        restore=True
        Ndat=N.data.copy(order='F')
    dpotri(Ndat,lower=lowapparent,overwrite_c=1)
    if restore:
        #copy data back
        N[()]=Ndat 
    N.attrs.update(cov_attrs()) 
    return N




def dtrsm_inplace(Chol:xr.DataArray,rhs:xr.DataArray,trans=0):
    """
    Triangular solve a RHS matrix with the Cholesky factor 
    """
    restore=False
    rhs_layout=memlayout(rhs.data)
    
    
    if rhs_layout == MemLayout.F_cont:
        rhsdat=rhs.data
        transa=trans
        side=0 #0 means left
    elif rhs_layout == MemLayout.C_cont:
        rhsdat=rhs.data.T
        transa=1-trans
        side=1 #1 means right  
    else:
        xinvlogger.warning("Right hand side matrix is not C or F contiguous, applying copy and restore")
        rhsdat=rhs.data.copy(order='F')
        transa=0
        side=1 #0 means left  
        restore=True

    lower=islower(Chol)
    chol_layout=memlayout(Chol.data)
    if chol_layout == MemLayout.C_cont:
        #To prevent an additional copy by f2py we can fake a F contigous array by transposing and switching lower/upper
        lower=1-lower
        transa=1-transa
        choldat=Chol.data.T
    else:
        choldat=Chol.data
    
    dtrsm(1.0,choldat,rhsdat,side=side,lower=lower,trans_a=transa,overwrite_b=1)

    if restore:    
        rhs.data[()]=rhsdat

    return rhs #although changed in place

def dsyrk_inplace(N,A,trans=0,beta=0.0,alpha=1.0):
    
    restore=False
    lower=islower(N)
    
    n_layout=memlayout(N.data)

    #check for C or F ordering
    if n_layout == MemLayout.F_cont:
        #fine F ordered array with stride 1 of double values
        #pointer
        Ndat=N.data
    elif n_layout == MemLayout.C_cont:
        #switch lower/upper representation in the F-ordered matrix of doubl
        lower=1-lower
        Ndat=N.data.T
    else:
        xinvlogger.warning("Normal matrix is not C or F contiguous, applying copy-restore")
        restore=True
        Ndat=N.data.copy(order='F')
    
    a_layout=memlayout(A.data)
    
    
    if a_layout == MemLayout.F_cont:
        adat=A.data
        transa=trans
    elif a_layout == MemLayout.C_cont:
        adat=A.data.T
        transa=1-trans
        side=1 #1 means right  
    else:
        xinvlogger.warning("Right hand side matrix is not C or F contiguous, applying copy and restore")
        adat=A.data.copy(order='F')
        transa=0
        restore=True
    dsyrk(alpha=alpha, a=adat,beta=beta, c=Ndat, trans=transa, lower=lower, overwrite_c=1)
    
    if restore:    
        A[()]=adat

    return N
