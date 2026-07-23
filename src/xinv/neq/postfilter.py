## Author: Kiana Karimi, s.karimi@utwente.nl July 2026

import numpy as np
import xarray as xr
import xinv
from xinv.linalg.inplace import cholesky_inplace,dpotri_inplace,dsymm_inplace
from xinv.core.attrs import islower
from scipy.linalg.blas import dgemm


def mirror_symmetric(M):

    M_full=M.copy()
    
    if islower(M):
        lower=np.tril(M_full.data)
        M_full.data=lower+np.tril(lower,k=-1).T

    else:
        upper=np.triu(M_full.data)
        M_full.data=upper+np.triu(upper,k=1).T

    return M_full
        


def createfilterW(alpha,grcneq_cond,neq_reg):

## Compute the posteriori regularization matrix (dense_filter_matrix)  W = (N + alpha Phi)^{-1} N


    # compute dscomb_inv^-1
    dscomb=grcneq_cond.xi.reg(neq_reg,alpha=alpha)

    #(N + alpha Phi)^{-1}
    dscomb_inv=dscomb.N.copy(deep=True)
    
    # factorize dscomb_inv as dscomb_inv = U.T @ U
    cholesky_inplace(dscomb_inv)
    # use the Cholesky factor U to compute dscomb_inv^-1
    dpotri_inplace(dscomb_inv)
   
    ## mirror dscomb_inv symmetric matrix
    dscomb_inv_full=mirror_symmetric(dscomb_inv)


    ## Compute (N + alpha Phi)^{-1} N

    # import pdb
    # pdb.set_trace()
    
    W=xr.zeros_like(dscomb_inv_full)

    W.data=dgemm(alpha=1.0,a=dscomb_inv_full.data,b=grcneq_cond.N.data,c=W.data,beta=0.0)
    
    # dsymm_inplace(A=dscomb_inv,B=grcneq_cond.N.data,C=W,alpha=1.0,beta=0.0)
    
    W=W.assign_coords({'n_':('nm_',W.n.data),'m_':('nm_', W.m.data)}) 
    W.name='dense_filter_matrix'

    
    return W