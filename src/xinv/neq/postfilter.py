## Author: Kiana Karimi, s.karimi@utwente.nl Sep 2026

import numpy as np
import xarray as xr
import xinv
from xinv.linalg.inplace import cholesky_inplace,dpotri_inplace,dsymm_inplace
from xinv.core.attrs import islower
from shxarray.core.sh_indexing import SHindexBase


# def mirror_symmetric(M):

#     M_full=M.copy()
    
#     if islower(M):
#         lower=np.tril(M_full.data)
#         M_full.data=lower+np.tril(lower,k=-1).T

#     else:
#         upper=np.triu(M_full.data)
#         M_full.data=upper+np.triu(upper,k=1).T

#     return M_full
        


def createfilterW(alpha,grcneq_cond,neq_reg):

    """ Compute the posteriori regularization matrix (dense_filter_matrix)
    W = (N + alpha Phi)^{-1} N """


    # compute dscomb_inv^-1
    dscomb=grcneq_cond.xi.reg(neq_reg,alpha=alpha)

    #(N + alpha Phi)^{-1}
    dscomb_inv=dscomb.N.copy(deep=True)
    
    # factorize dscomb_inv as dscomb_inv = U.T @ U
    cholesky_inplace(dscomb_inv)
    
    # use the Cholesky factor U to compute dscomb_inv^-1
    dpotri_inplace(dscomb_inv)
   
    ## mirror dscomb_inv symmetric matrix
    # dscomb_inv_full=mirror_symmetric(dscomb_inv)


    ## Compute (N + alpha Phi)^{-1} N
    
    W=xr.zeros_like(grcneq_cond.N)
    
    dsymm_inplace(A=dscomb_inv,B=grcneq_cond.N,C=W,alpha=1.0,beta=0.0)

    W=W.sh.set_nmindex(SHindexBase.mi_toggle(W.indexes[SHindexBase.name]),suf='_')
    
    return W