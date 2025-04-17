## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl


from scipy.linalg import cholesky
from scipy.linalg.blas import dtrsm, dsyrk # import dsyrk
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
    """ merge two normal equation systems"""
    
    #N1,rhs1,ltpl1,sigma01,nobs1,npara1=find_neq_components(dsneq)
    #N2,rhs2,ltpl2,sigma02,nobs2,npara2=find_neq_components(dsneqother)
    
    if set(dsneq.variables) !=set(dsneqother.variables):
        raise TypeError("Datasets should have the same variables")
    if set(dsneq.dims) != set(dsneqother.dims):
        raise TypeError("Datasets should have the same dimensions")
    
#     if not dsneq.equals(dsneqother):
#         raise TypeError("Datasets are not identical and cannot be merged")
     
    if "fingerprints" not in dsneq.dims:
        dsneq=dsneq.expand_dims("fingerprints")
    if "fingerprints" not in dsneqother.dims:
        dsneqother=dsneqother.expand_dims("fingerprints")
                                     
    if "N" in dsneq and "N" in dsneqother:
        Nnew=xr.concat([dsneq.N,dsneqother.N],dim="fingerprints") # fingerprints refers to the characteristic spatial patterns of mass change such as ice sheet, glaciers, TWS and GIA changes
    else:
        raise TypeError("Both datasets should have N")

    if "rhs" in dsneq and "rhs" in dsneqother:
        rhsnew=xr.concat([dsneq.rhs,dsneqother.rhs],dim="fingerprints")
    else:
        raise TypeError("Both datasets should have rhs")
        
    if "ltpl" in dsneq and "ltpl" in dsneqother:
        ltplnew=xr.concat([dsneq.ltpl,dsneqother.ltpl],dim="fingerprints")
    else:
        raise TypeError("Both datasets should have ltpl")
        
    order='C'
    coords={**dsneq.coords,**dsneqother.coords}
    dsneq_merged=xr.Dataset(data_vars=dict(N=(Nnew.dims,Nnew.data),rhs=(rhsnew.dims,rhsnew.data),ltpl=(ltplnew.dims,ltplnew.data)),coords=coords)

        
    return dsneq_merged

    
    
def merge(fwdoperator,fwdoperatorother):
    """ Merge two design matrices that are composed of the corresponding fingerprints"""
    
    ## create a new coordinate that represents the number of basins
    
    
    
    Basin_fgprn1=[str(x) for x in fwdoperator.coords["basinid"].values]
    basin_fgprn2=[str(x) for x in fwdoperatorother.coords["basins"].values]
    
    basinsnum=np.concatenate([Basin_fgprn1,basin_fgprn2])

    fwdoperator, fwdoperatorother = xr.align(fwdoperator, fwdoperatorother, join="outer", fill_value=0)

    jac_basins=np.hstack([fwdoperator.values,fwdoperatorother.values])
    
    dim=fwdoperatorother.dims
    mrgd_fwdoperator=xr.DataArray(jac_basins,dims=dim,coords=dict(nm=fwdoperator.coords["nm"],basins=basinsnum))

    return mrgd_fwdoperator
    
    
def transform(dsneq:xr.Dataset, fwdoperator):
    """ Transform the normal equation system using a forward operator matrix (e.g., the decorrelated jacobian matrix)"""
    
    # if not inplace:
    #     # copy entire NEQ and operate on that one in place
    #     dsneq=dsneq.copy(deep=True)
    
    N,rhs,ltpl,sigma0,nobs,npara=find_neq_components(dsneq)
        
    trns_param=fwdoperator.shape[0] #number of transformed parameters
    
    Nnew=np.zeros((trns_param,trns_param))
    rhsnew=np.zeros((trns_param,1))
    ltplnew=np.zeros((trns_param,))
    
    for i in range(prd):
        Nnew[i]=dsyrk(1.0,fwdoperator[i],trans=1,lower=1)
        rhsnew[i]=(fwdoperator.T@rhs).reshape(-1,1) # replace with xinv * later
        ltplnew[i]=ltpl[i]  ## add an if condition for changing the ltpl if set_apriori changes
        
    
    # from xinv.core.attrs import ltplnew_attrs,Nnew_attrs,rhsnew_attrs
    # N.attrs.update(Nnew_attrs())
    # ltpl.attrs.update(ltplnew_attrs())
    # rhs.attrs.update(rhsnew_attrs())

    dsneq["N"]=Nnew
    dsneq["rhs"]=rhsnew
    dsneq["ltpl"]=ltplnew
    
#    dsneq=dsneq.rename(dict(N='Nnew',rhs='rhsnew',ltpl='ltplnew'))
    
    
    if inplace:
        return None
    else:
        return dsneq



def scaling_factor(dsneq:xr.Dataset,fwdoperator):
    """ This function returns TWS in Gton by solving the normal equation system"""
    N,rhs,ltpl,sigma0,nobs,npara=find_neq_components(dsneq)
    
    prd=N.shape[0]
    x=np.zeros((prd,fwdoperator.shape[1],1))
    
    for i in range(prd):
        x[i] = solve_triangular(N[i], rhs[i], trans = 'N', lower=True) ## fix the lower-upper later
            
    return np.squeeze(x)