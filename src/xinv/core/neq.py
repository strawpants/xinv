## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl


from scipy.linalg import cholesky
from scipy.linalg.blas import dtrsm
from scipy.linalg.lapack import dpotri
from xinv.core.attrs import cov_attrs,solest_attrs,ltpl_attrs,find_neq_components,sigma0_attrs,Chol_attrs
from xinv.core.exceptions import XinvIllposedError

import numpy as np
import xarray as xr
def solve(dsneq,inplace=False):
    """Solve the normal equation system using Cholesky factorization"""
    
    if not inplace:
        #copy entire NEQ and operate on that one in place
        dsneq=dsneq.copy(deep=True)


    N,rhs,ltpl,sigma0,nobs,npara=find_neq_components(dsneq)
    #decompose the normal matrix using cholesky in place
    if N.attrs['xinv_state'] == 'SymUpper':
        lower=0
    elif N.attrs['xinv_state'] == 'SymLower':
        lower=1
    else:
        raise RuntimeError(f"Don't know (yet) how to cope with normal matrix state:{N.attrs['xinv_state']}")

    
    #check for C or F ordering
    if N.data.flags['F_CONTIGUOUS'] and N.data.strides[0] == 8:
        #fine F ordered array with stride 1 of double values
        lowapparent=lower
    elif N.data.flags['C_CONTIGUOUS'] and N.data.strides[1] == 8:
        #switch lower/upper representation in the F-ordered matrix of doubl
        lowapparent=1-lower
    else:
        raise RuntimeError("Normal matrix is not C or F contiguous")
    try:
        cholesky(N.data,lower=lowapparent,overwrite_a=1)
    except np.linalg.LinAlgError as e:
        raise XinvIllposedError(str(e))
    N.attrs.update(Chol_attrs(lower))
    
    #Solve the system in several steps
    #1. triangular solve U'z=rhs for z (U' is now the Cholesky factor of N)
    #dtrsm(alpha, a, b[, side, lower, trans_a, diag, overwrite_b]) 
    #breakpoint()
    
    if rhs.data.flags['F_CONTIGUOUS'] and rhs.data.strides[0] ==8:
        rhs_is_c_cont=1
    elif rhs.data.flags['C_CONTIGUOUS'] and rhs.data.strides[1] == 8:
        rhs_is_c_cont=0
    else:
        raise RuntimeError("Right hand side matrix is not C or F contiguous")

    
    if rhs_is_c_cont:
        transa=1
        side=0 #0 means left
    
        dtrsm(1.0,N.data,rhs.data,side=side,lower=lowapparent,trans_a=transa,overwrite_b=1)
    else:
        #rhs is F contigious
        transa=0
        side=1 #0 means left  
        dtrsm(1.0,N.data,rhs.data.T,side=side,lower=lowapparent,trans_a=transa,overwrite_b=1)

        
    #2. update ltpl
    
    ltpl-=rhs.dot(rhs,dim=N.dims[0])
    ltpl.attrs.update(ltpl_attrs('posteriori'))

    #3 solve Ux=z for x
    
    
    #call dtrsv('U','N','N',n1,C,n1,d,1) !d is updated again
    if rhs_is_c_cont:
        transa=0
        side=0 #0 means left
        dtrsm(1.0,N.data,rhs.data,side=side,lower=lowapparent,trans_a=transa,overwrite_b=1)
    else:
        transa=1
        side=1 #0 means left
        dtrsm(1.0,N.data,rhs.data.T,side=side,lower=lowapparent,trans_a=transa,overwrite_b=1)
    
    # dtrsv(N.data,rhs.data,lower=0,trans=0,overwrite_x=1)
    rhs.attrs.update(solest_attrs())


    #4 compute error-covariance 
    # call dpotri('U',n1,C,n1,info)
    dpotri(N.data,lower=lowapparent,overwrite_c=1)
    N.attrs.update(cov_attrs()) 

    # compute posteriori sigma0
    sigma0=np.sqrt(ltpl/(nobs-npara))
    sigma0.attrs.update(sigma0_attrs('posteriori'))
    dsneq['sigma0']=sigma0
    
    dsneq=dsneq.rename(dict(N='COV',rhs='solution'))
    if inplace:
        return None
    else:
        return dsneq

    
def reduce(dsneq:xr.Dataset, idx):
    #tbd reduce (implicitly solve) variables from a normal equation system spanned by idx
    raise NotImplementedError("Reduce operation not yet implemented")

def fix(dsneq:xr.Dataset, idx):
    #tbd fix and (remove) solve) variables from a normal equation system spanned by idx
    raise NotImplementedError("Fix operation not yet implemented")

def set_apriori(dsneq:xr.Dataset, daapri:xr.DataArray):
    #tbd set/change apriori values in a normal equation system
    raise NotImplementedError("Set apriori values not yet implemented")


def add(dsneq:xr.Dataset, dsneqother:xr.Dataset):
    #tbd add/merge two normal equation systems
    raise NotImplementedError("Adding/merging NEQS not yet implemented")

def transform(dsneq:xr.Dataset, fwdoperator):
    #tbd transform normal equation system using a forward transformation matrix
    raise NotImplementedError("Transforming NEQS not yet implemented")
