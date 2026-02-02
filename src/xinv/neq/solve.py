## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

from scipy.linalg import cholesky
import numpy as np
import xarray as xr

from scipy.linalg.blas import dtrsm
from scipy.linalg.lapack import dpotri
from xinv.core.exceptions import XinvIllposedError

from xinv.core.attrs import cov_attrs, solest_attrs,ltpl_attrs,find_neq_components,sigma0_attrs,Chol_attrs,xinv_st,islower

from xinv.linalg.inplace import cholesky_inplace, dpotri_inplace,dtrsm_inplace

def solve(dsneq,inplace=False):
    """Solve the normal equation system using Cholesky factorization"""
    
    if not inplace:
        #copy entire NEQ and operate on that one in place
        dsneq=dsneq.xi.deepcopy(order='F')
        #dsneq=dsneq.copy(deep=True)


    N,rhs,x0,ltpl,sigma0,nobs,npara=find_neq_components(dsneq)
    
    # if not inplace:
        # #note xarray copy does not always do actual deep copy
        # rhs.data=rhs.data.copy(order='F')
        # #link back to output neq
        # dsneq[rhs.name]=rhs

        # #note xarray copy does not always do actual deep copy
        # N.data=N.data.copy(order='F')
        # #link back to output neq
        # dsneq[N.name]=N

    b=rhs.xi.deepcopy()
    #b.data=rhs.data.copy(order='F')    
    #decompose the normal matrix using cholesky in place N -> U'U or LL'
    cholesky_inplace(N)
    
    
    #Solve the system in several steps
    #1. triangular solve U'z=rhs for z (U' is now the Cholesky factor of N)
   
    dtrsm_inplace(N,rhs,trans=1)

    #2 solve Ux=z for x
    
    dtrsm_inplace(N,rhs)
    
    #3. update ltpl
    ltpl-=rhs.dot(b,dim=N.dims[0])
    ltpl.attrs.update(ltpl_attrs('posteriori'))
    
    # dtrsv(N.data,rhs.data,lower=0,trans=0,overwrite_x=1)
    rhs.attrs.update(solest_attrs())


    #4 compute error-covariance 
    # call dpotri('U',n1,C,n1,info)
    dpotri_inplace(N)
    # compute posteriori sigma0
    
    sigma0=np.sqrt(ltpl/(nobs-npara))
    sigma0.attrs.update(sigma0_attrs('posteriori'))
    dsneq['sigma0']=sigma0
    
    dsneq=dsneq.rename(dict(N='COV',rhs='solution'))
    if inplace:
        return None
    else:
        return dsneq
