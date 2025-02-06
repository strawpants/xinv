## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl


from scipy.linalg import cholesky
from scipy.linalg.blas import dtrsm
from scipy.linalg.lapack import dpotri
from xinv.core.attrs import cov_attrs,solest_attrs,ltpl_attrs,find_neq_components,sigma0_attrs,Chol_attrs
import numpy as np

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

    cholesky(N.data,lower=lowapparent,overwrite_a=1)
    N.attrs.update(Chol_attrs(lower))
    
    #Solve the system in several steps
    #1. triangular solve U'z=rhs for z (U' is now the Cholesky factor of N)
    #dtrsm(alpha, a, b[, side, lower, trans_a, diag, overwrite_b]) 
    transa=1
    side=0 #0 means left
    dtrsm(1.0,N.data,rhs.data,side=side,lower=lower,trans_a=transa,overwrite_b=1)

    #2. update ltpl
    ltpl-=rhs.dot(rhs,dim=N.dims[0])
    ltpl.attrs.update(ltpl_attrs('posteriori'))

    #3 solve Ux=z for x
    #call dtrsv('U','N','N',n1,C,n1,d,1) !d is updated again
    transa=0
    side=0 #0 means left
    dtrsm(1.0,N.data,rhs.data,side=side,lower=lower,trans_a=transa,overwrite_b=1)
    # dtrsv(N.data,rhs.data,lower=0,trans=0,overwrite_x=1)
    rhs.attrs.update(solest_attrs())


    #4 compute error-covariance 
    # call dpotri('U',n1,C,n1,info)
    dpotri(N.data,lower=lower,overwrite_c=1)
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

    
def reduce(dsneq, idx):
    #tbd reduce (implicitly solve) variables from a normal equation system spanned by idx
    raise NotImplementedError("Reduce operation not yet implemented")


# def fix(self,variables=None):
    # """Fix the indicated variables to its a apriori values"""
    # return self._obj

# def set_apriori(self,da):
    # """Change the a priori values of the system"""
    # return self._obj


# def add(self,other):
    # """Add one normal equation system to another. Common parameters will be added, the system will be extended with unique parameters of the other system"""
    # return self._obj

