## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl


from scipy.linalg import cholesky
from scipy.linalg.blas import dtrsm
from scipy.linalg.lapack import dpotri
from scipy.linalg import cholesky
from xinv.core.attrs import cov_attrs, find_component, find_xinv_coords,solest_attrs,ltpl_attrs,find_neq_components,sigma0_attrs,Chol_attrs,N_attrs,rhs_attrs,nobs_attrs,npara_attrs, xunk_coords_attrs,xinv_tp,xinv_st
from xinv.core.grouping import find_group_coords,build_group_coord
from xinv.core.exceptions import XinvIllposedError
from xinv.core.logging import xinvlogger
from xinv.core.tools import find_ilocs

import numpy as np
import xarray as xr
import pandas as pd

def zeros(rhsdims,coords,lower=0):
        
        
        #figure out the shape of the rhs from the provided coordinates
        rhsshape=[len(coords[dim]) for dim in rhsdims]

        #Note: We assume that the first dimension is the unknown parameter dimension
        if not "xinv_type" in coords[rhsdims[0]].attrs:
            xinvlogger.warning("Unknown parameter dimension is assumed to be the first one {rhsdims[0]}")
        elif coords[rhsdims[0]].attrs["xinv_type"] != xinv_tp.unk_co:
            raise ValueError("The unknown parameter does not correspond to the first dimenion of the right hand side")

        #allocate space
        N=([rhsdims[0],rhsdims[0]+'_'],np.zeros([rhsshape[0],rhsshape[0]]))
        rhs=(rhsdims,np.zeros(rhsshape))
        ltpl=(rhsdims[1:],np.zeros(rhsshape[1:]))
        sigma0=(rhsdims[1:],np.zeros(rhsshape[1:]))
        nobs=(rhsdims[1:],np.zeros(rhsshape[1:],dtype=np.int64))
        npara=(rhsdims[1:],np.zeros(rhsshape[1:],dtype=np.int64))

        dsneq=xr.Dataset(dict(N=N,rhs=rhs,ltpl=ltpl,nobs=nobs,npara=npara,sigma0=sigma0),coords=coords)
        #add attributes
        dsneq.N.attrs.update(N_attrs(lower))
        dsneq.rhs.attrs.update(rhs_attrs())
        dsneq.ltpl.attrs.update(ltpl_attrs('apriori'))
        dsneq.sigma0.attrs.update(sigma0_attrs('apriori'))
        dsneq.nobs.attrs.update(nobs_attrs())
        dsneq.npara.attrs.update(npara_attrs())

        return dsneq

def solve(dsneq,inplace=False):
    """Solve the normal equation system using Cholesky factorization"""
    
    if not inplace:
        #copy entire NEQ and operate on that one in place
        dsneq=dsneq.copy(deep=True)


    N,rhs,ltpl,sigma0,nobs,npara=find_neq_components(dsneq)

    b=rhs.copy(deep=True)

    
    #decompose the normal matrix using cholesky in place
    if N.attrs['xinv_state'] == xinv_st.symU:
        lower=0
    elif N.attrs['xinv_state'] == xinv_st.symL:
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
    
    ltpl-=rhs.dot(b,dim=N.dims[0])
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
    try:
        N1,rhs1,ltpl1,sigma01,nobs1,npara1=find_neq_components(dsneq)
        N2,rhs2,ltpl2,sigma02,nobs2,npara2=find_neq_components(dsneqother)
    except KeyError:
        raise RuntimeError("Cannot find all NEQ components, are the appropriate xinv_ attributes set?")
    
    if N1.name != N2.name or rhs1.name != rhs2.name or ltpl1.name != ltpl2.name or sigma01.name != sigma02.name or nobs1.name != nobs2.name or npara1.name != npara2.name:
        xinvlogger.warning("Normal system variables have inconsistent variable names, using names from the first system")

    #check for uniform sigma scaling and raise an error if not
    sigma_ratio=sigma01/sigma02
    if len(sigma_ratio.shape) == 0:
        sigma_ratio_single=sigma_ratio.item()
        hasauxdim=False 
    else:
        sigma_ratio_single=sigma_ratio[0].item()
        if not (sigma_ratio == sigma_ratio_single).all():
            raise ValueError(f"Merging NEQs with a common normal matrix but with varying sigma0 ratios over the auxiliary dimensions is not possible")
        hasauxdim=True


    var2_ratio=sigma_ratio_single**2
    #check matrix compatibility
    #check if N is in upper triangular or lower triangular form
    if N1.attrs['xinv_state']!=N2.attrs['xinv_state']:
        raise RuntimeError(f"Can't currently merge NEQs with different xinv_state:{N1.attrs['xinv_state']} vs {N2.attrs['xinv_state']}")
    
    unkdim1=N1.dims[0]
    unkdim2=N2.dims[0]
    
    #find out the unique unknown parameters
    # unique_unk_coord=pd.MultiIndex.from_tuples(np.unique(np.concatenate([N1[unkdim1].data,N2[unkdim2]])))
    unique_unk_coord=xr.DataArray(np.unique(np.concatenate([N1[unkdim1].data,N2[unkdim2]])),dims=[unkdim1],name=unkdim1)
    unique_unk_coord.attrs.update(xunk_coords_attrs(state=xinv_st.linked))

    #find the auxiliary dimensions (ignore the unknown parameter dimension, and group_id/and seq)
    
    xinvcoords=find_xinv_coords(dsneq,exclude=[xinv_tp.grp_id_co,xinv_tp.grp_seq_co])
    group_id_co,group_seq_co=find_group_coords(dsneq)

    #check if the group id and sequence coordinates are present in the first system
    if group_id_co is not None and group_seq_co is not None:
        #turn the unknown coordinate into a multiindex
        grp_co=build_group_coord(unique_unk_coord,dim=unkdim1,group_id_name=group_id_co.name,group_seq_name=group_seq_co.name)
        xinvcoords[unkdim1]=grp_co[unkdim1]
        xinvcoords[group_id_co.name]=grp_co[group_id_co.name]
        xinvcoords[group_seq_co.name]=grp_co[group_seq_co.name]
    else:

        #replace the unknow coordinate with the union version
        xinvcoords[unkdim1]=unique_unk_coord
    #add the proper attributes

    
    
    
    xinvcoordsother=find_xinv_coords(dsneqother,exclude=[xinv_tp.grp_id_co,xinv_tp.grp_seq_co])
    #possibly update with complementary coordinates from the second system
    xinvcoords.update({ky:coord for ky,coord in xinvcoordsother.items() if ky not in xinvcoords.keys()})
    #allocate space for the combined output normal equation system and use system one as the base template
    dsneq_merged=xr.Dataset.xi.neqzeros(rhsdims=rhs1.dims,coords=xinvcoords)
    #for some reason the multindex coordinate attributes do not get properly propagated
    #so make sure they are added
    for key,coord in xinvcoords.items():
        if not dsneq_merged[key].attrs:
            dsneq_merged[key].attrs.update(coord.attrs)

    #rename the variables to the same name as in the first system
    renamedict=dict(N=N1.name,rhs=rhs1.name,ltpl=ltpl1.name,sigma0=sigma01.name,nobs=nobs1.name,npara=npara1.name)
    dsneq_merged=dsneq_merged.rename(renamedict)
    
    #get links to the combined normal equation system
    Nout,rhsout,ltplout,sigma0out,nobsout,nparaout=find_neq_components(dsneq_merged)
    #Note: we need to be very careful when assigning values to the above views so that we insert data at the existing memory locations (!!), rather than replace the views alltogether
    #add stuff from the first normal equation system
    
    idx1=find_ilocs(dsneq_merged,unkdim1,N1[unkdim1].data)
    Nout[idx1,idx1]=N1.data
    rhsout[{unkdim1:idx1}]=rhs1.data
    
    #note use [()] to replace the value in place
    ltplout[()]=ltpl1
    sigma0out[()]=sigma01
    nobsout[()]=nobs1
    nparaout[()]=np.int64(dsneq_merged.sizes[unkdim1])
    #possibly add reduced parameters from the first system
    nreduced=npara1-dsneq.sizes[unkdim1]
    nparaout+=nreduced.astype(np.int64)


    
    #add stuff from the second normal equation system
    idx2=find_ilocs(dsneq_merged,unkdim2,N2[unkdim2].data)

    
    
    #Note that we need to take care of possible different sigma0's
    if var2_ratio != 1.0:
        xinvlogger.info(f"Applying a rescaling of {var2_ratio} to the second normal equation system to compensate for the different apriori variances")
    
    Nout[idx2,idx2]+=var2_ratio*N2.data
    
    rhsout[{unkdim1:idx2}]+=var2_ratio*rhs2.data
    ltplout+=var2_ratio*ltpl2
    #note dsneq_merged['sigma0'] stays the same 
    nobsout+=nobs2
    #possibly add the amount of reduced parameters from the second system
    nreduced=npara2-dsneqother.sizes[unkdim2]
    nparaout+=nreduced.astype(np.int64)

    
    return dsneq_merged                     
    

def transform(dsneq:xr.Dataset,fwdoperator:xr.DataArray,lower=0):
    
    """ Transform a normal equation system using a forward operator"""

    if dsneq.rhs.attrs["xinv_type"]=="aprioriVec":
        dsneq.rhs.attrs.update(rhs_attrs())

    if "sigma0" not in dsneq:
        dsneq["sigma0"]=xr.DataArray(1.0)
        dsneq.sigma0.attrs.update(sigma0_attrs())
    
    N,rhs,ltpl,sigma0,nobs,npara=find_neq_components(dsneq)
    
    unkdim=N.dims[0]

    if fwdoperator.dims[0] != unkdim:
        raise ValueError("fwdoperator last dimension must match the unknown dimension")

    rhs_transformed=xr.dot(fwdoperator.transpose(),rhs,dim=unkdim)

    
    Ncholesky=cholesky(N.data,lower=lower)
    decorrfwdoperator=Ncholesky@fwdoperator.data
    decorrfwdoperator=xr.DataArray(decorrfwdoperator,dims=fwdoperator.dims,coords=fwdoperator.coords)

    N_transformed=decorrfwdoperator.transpose().data@decorrfwdoperator.data
    
    newdim=fwdoperator.dims[1]
    nnew=fwdoperator.sizes[newdim]

   # new_unk=xr.DataArray(np.arange(nnew),dims=newdim,name=newdim)
    new_unk=fwdoperator.coords["xinv_unk"]   
    #"xinv_unk_": fwdoperator.coords["xinv_unk"]
    new_unk.attrs.update(xunk_coords_attrs(state=xinv_st.linked))
 
    #coords=dict(find_xinv_coords(dsneq,exclude=[xinv_tp.grp_id_co,xinv_tp.grp_seq_co]))
    coords={newdim:new_unk}
    
    N_transformed=xr.DataArray(N_transformed,dims=(newdim,newdim+"_"),coords=coords)

    #import pdb; pdb.set_trace()
    npara=xr.DataArray(nnew,attrs=npara_attrs(),name="npara")

    sigma_trans=np.sqrt(ltpl/(nobs-npara.data))
    sigma_trans.attrs.update(sigma0.attrs)


    
    dsneq_trans=xr.Dataset.xi.neqzeros(rhsdims=rhs_transformed.dims,coords=coords,lower=lower)
    renamedict=dict(N=N.name,rhs=rhs.name,ltpl=ltpl.name,sigma0=sigma0.name,nobs=nobs.name,npara=npara.name)
    dsneq_trans=dsneq_trans.rename(renamedict)
    
    dsneq_trans["N"]=N_transformed
    dsneq_trans.N.attrs.update(N_attrs(lower=lower))

    dsneq_trans["rhs"]=rhs_transformed
    dsneq_trans.rhs.attrs.update(rhs.attrs)

    dsneq_trans["ltpl"]=ltpl
    dsneq_trans.ltpl.attrs.update(ltpl.attrs)

    dsneq_trans["sigma0"]=sigma_trans

    dsneq_trans["nobs"]=nobs
    dsneq_trans.nobs.attrs.update(nobs.attrs)

    dsneq_trans["npara"]=npara
    dsneq_trans.npara.attrs.update(npara.attrs)

    return dsneq_trans



    


    
    