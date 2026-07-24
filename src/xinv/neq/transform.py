## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import xarray as xr
from xinv.core.attrs import find_component, get_xunk_size_coname,xinv_tp,islower,find_neq_components,unlink,find_xinv_coords,get_type,xunk_coords_attrs,xinv_st,N_attrs,sigma0_attrs,rhs_attrs,ltpl_attrs,nobs_attrs,npara_attrs
from xinv.neq.neq import zeros as neqzeros
from xinv.core.tools import find_overlap,find_ilocs2,unique_union
from xinv.fwd.matrixfwd import MatrixfwdOp
import numpy as np

from scipy.linalg.blas import dtrmm,dsyr2k
from scipy.linalg.lapack import dgesv

def transform(dsneq:xr.Dataset,fwdoperator,apriori_strategy="ignore",**kwargs):
    """Transform a normal equation system using a forward operator

    Parameters
    ----------
    dsneq : xr.Dataset
        The normal equation system to be transformed
    fwdoperator : xinv.fwd.FwdOperator or xarray.DataArray
        The forward operator to transform the normal equation system. This can also be a xarray.DataArray holding the Design matrix
    apriori_strategy : str, optional
        How to treat the transformation of the apriori solution estimate, currently only "ignore" is allowed (sets the output to zero without changing the ltpl)
    
    **kwargs :
        Additional arguments passed to the fwdoperator.jacobian call
    """
   



    if type(fwdoperator) == xr.DataArray:
        # Convenience service: convert the design matrix to a linear forward operator
        fwdoperator=MatrixfwdOp(fwdoperator)
    
    if apriori_strategy not in ["ignore"]:
        raise NotImplementedError(f"apriori strategy {apriori_strategy} not implemented, only 'ignore' is currently allowed")

    #input unknown size and dimension name
    i_unksz,i_unkdim=get_xunk_size_coname(dsneq)

    if fwdoperator.obs_dim != i_unkdim:
        raise ValueError(f"fwdoperator row dimension name {fwdoperator.obs_dim} must match the unknown dimension {i_unkdim}")

    #retrieve the forward operator Jacobian (note: may not be linear, and may need the current apriori solution input to generate the Jacobian)

    #pointers to the input normal equation system
    i_N,i_rhs,i_x0,i_ltpl,i_sigma0,i_nobs,i_npara=find_neq_components(dsneq)
    
    if i_x0 is None:
        i_x0=xr.zeros_like(i_rhs)

    #jac=fwdoperator.jacobian(daobs=i_x0,**kwargs)#.reset_index(i_unkdim)
    jac=fwdoperator.jacobian(daobs=dsneq,**kwargs)#.reset_index(i_unkdim)
    
    #find unique and overlapping coordinates over the transform dimension
    jauniq,common,neuniq=find_overlap(jac.get_index(i_unkdim),dsneq.get_index(i_unkdim))
    # jauniq,common,neuniq=find_overlap_coords(jac[i_unkdim],dsneq[i_unkdim])
    if len(jauniq) > 0:
        raise ValueError(f"Forward operator has unknown coordinate values {jauniq} which are not found in the input normal equation system {dsneq[i_unkdim]}")
    
    #needed so the rest of the operations using the jacobian does not get messed up with potential name-conflicting indices

    jac=jac.reset_index(i_unkdim)
    #index vector of the input to transformed parameters
    idxtrans=find_ilocs2(dsneq.get_index(i_unkdim),common)
    idxnotrans=None 
    if len(neuniq) == 0:
        partial=False
        #full transform, all NEQ unknowns are transformed 
        o_unkdim=fwdoperator.unkdim
        outcoords={o_unkdim:jac[o_unkdim]}

    else:
        # this is more complex as some parameters will be removed/introduced, but some remain in the new syste
        partial=True
        o_unkdim='xinv_unk'
        o_tdim=fwdoperator.unkdim

        midx=xr.Coordinates.from_pandas_multiindex(unique_union(neuniq,jac.get_index(o_tdim)),o_unkdim)
        outcoords={o_unkdim:midx}


    #possibly add coordinates from the Jacobian which are associated with a group coordinate
     
    #add auxiliary coordinates from the input rhs
    for k,v in i_rhs.coords.items():
        if not bool(v.dims) or (v.dims[0] != i_unkdim):
            outcoords[k]=v

    #Allocate the new output normal equations system
    #setup dimensions of the output system
    o_rhsdims=[o_unkdim]
    
    o_rhsdims.extend([dname for dname in i_rhs.dims if dname != i_unkdim])
    
    i_lower=islower(i_N)
    o_dsneq= neqzeros(rhsdims=o_rhsdims,coords=outcoords,lower=i_lower)
    #xexplicitly link new unknown coordinate (not always carried over from Jacobian)
    o_dsneq[o_unkdim].attrs.update(xunk_coords_attrs(xinv_st.linked))
    #pointers to the new normal equation system
    o_N,o_rhs,o_x0,o_ltpl,o_sigma0,o_nobs,o_npara=find_neq_components(o_dsneq)

    #setup slicing
    if partial:
        o_tsz=jac.sizes[o_tdim]
        o_unksz=o_rhs.sizes[o_unkdim]
        #slice representing the untransformed part of the unknowns
        uslice=slice(0,o_unksz-o_tsz)
        #slice representing the transformed part of the unknowns
        tslice=slice(o_unksz-o_tsz,None)
    else:
        o_unksz=o_rhs.sizes[o_unkdim]
        #no untransformed part
        uslice=None
        #transformed part is a complete slice
        tslice=slice(None)
        
    #transformed part (valid for both partial and non-partial transforms
    #right hand side
    
    #unpack jacobian if it is sparse (todo use sparse triangular routines once they become available)
    if hasattr(jac.jacobian.data,"todense"):
        jac.jacobian.data=jac.jacobian.data.todense()
    # add the transformed part of the input rhs to the output rhs (note: inplace operation in o_rhs)
    
    o_rhs[{o_unkdim:tslice}]=xr.dot(jac.jacobian,i_rhs.isel({i_unkdim:idxtrans}),dim=i_unkdim).data

    #transform normal matrix

        
    #Ntt ndarray will hold the transformed part of the normal matrix
        
    o_Ntt=o_N[{o_unkdim:tslice,o_unkdim+'_':tslice}].data
        
    #Decompose the input Normal matrix  as U^t + U 
    
    # we can apply the following trick: decompose the input matrix i_Ntt in an upper triangular matrix U so
    # i_Ntt = U + U^T, U is the upper triangle of i_Ntt but with the diagonal divided by 2
    
    # The transformation can be written as
    # o_Ntt = A^T (i_Ntt) A  = A^T (U + U^T)  A =  A^T ( U A )  + (A^T U^T) A
    # when UA is written as X we can decompose the entire transform as 
    # (1) extract the triangular matrix U = triu(N) and divide the diagonal by 2
    # (2) compute the transformed matrix as X = U A
    # (3) apply a dsyr2k operation as  Nnew = A^T X + X^T A
        
    #get the index vector to sort N according to the forward operator obs_dim
    if i_lower:
        #extract and mirror lower triangular matrix (mirroring is needed because a resorting may move elements from upper to lower triangle)
        U=np.tril(i_N.data, k=-1) + np.tril(i_N.data, k=-1).T
    else:
        #extract and mirror upper triangular matrix
        U=np.triu(i_N.data, k=1) + np.triu(i_N.data, k=1).T
        
    np.fill_diagonal(U, i_N.data.diagonal()/2)
        
    #resort 
    
    if partial:
        #in case of a partial transform we also need get the cross sectional matrix
        # index of parameters to keep and transform
        idxkeep=find_ilocs2(dsneq.get_index(i_unkdim),neuniq)
        if type(idxtrans) == slice and type(idxkeep) == slice:
            i_Ntu=U[idxtrans,idxkeep]
        elif type(idxtrans) == slice:
            
            i_Ntu=U[np.ix_(np.arange(idxtrans.start,idxtrans.stop+1,idxtrans.step),idxkeep)]
        elif type(idxkeep) == slice:
            i_Ntu=U[np.ix_(idxtrans,np.arange(idxkeep.start,idxkeep.stop+1,idxkeep.step))]
        else:
            i_Ntu=U[np.ix_(idxtrans,idxkeep)]

    if type(idxtrans) == slice:
        U=U[idxtrans,idxtrans]
    else:
        U=U[np.ix_(idxtrans,idxtrans)]
        
    # retrieve X by a triangular matrix multiplication
    # subroutine dtrmm 	( 	character 	side,
            # character 	uplo,
            # character 	transa,
            # character 	diag,
            # integer 	m,
            # integer 	n,
            # double precision 	alpha,
            # double precision, dimension(lda,*) 	a,
            # integer 	lda,
            # double precision, dimension(ldb,*) 	b,
            # integer 	ldb )
    X=dtrmm(1.0,U,jac.jacobian.data)
       
    # now apply the dsyr2k operation to compute the transformed normal matrixa

        # dsyr2k(alpha, a, b[, beta, c, trans, lower, overwrite_c]) = <fortran function dsyr2k>
    trans=1
    #TODO: avoid memory copy of the output subslice matrix by calling blas from cython (thereby avoiding f2py which makes intermediate copies when passing subarrays)
    o_Ntt[()]=dsyr2k(1.0,jac.jacobian.data,X,trans=trans,lower=i_lower)


    if partial:
        #we need to process the cross terms and copy the untransformed part

        
        #copy the untransformed part of the input normal matrix
        if type(idxkeep) == slice:
            o_N[{o_unkdim:uslice,o_unkdim+'_':uslice}]=i_N.data[idxkeep,idxkeep]
        else:
            o_N[{o_unkdim:uslice,o_unkdim+'_':uslice}]=i_N.data[np.ix_(idxkeep,idxkeep)]
        
        #copy right hand side
        o_rhs[{o_unkdim:uslice}]=i_rhs[{i_unkdim:idxkeep}].data

        #copy apriori values
        o_x0[{o_unkdim:uslice}]=i_x0[{i_unkdim:idxkeep}].data
        # breakpoint()

        if i_lower == 0:
            #fill upper cross sectional nromal matrix part
            o_N[{o_unkdim:uslice,o_unkdim+"_":tslice}]=i_Ntu.T@jac.jacobian.data
        else:
            #fill upper cross sectional area of the normal matrix
            o_N[{o_unkdim:tslice,o_unkdim+"_":uslice}]=jac.jacobian.T.data@i_Ntu 




    # copy ltpl's fromt he old system
    o_ltpl[()]=i_ltpl
    o_sigma0[()]=i_sigma0
    o_nobs[()]=i_nobs
    o_npara[()]=i_npara
    #Note we may need to adjust the amount of parameters because some may have been removed or newly introduced
    
    o_npara[()]+=o_unksz-i_unksz 
    o_N.attrs.update(N_attrs(i_lower))
    o_rhs.attrs.update(rhs_attrs())
    o_ltpl.attrs.update(ltpl_attrs('apriori'))
    o_sigma0.attrs.update(sigma0_attrs('apriori'))
    o_nobs.attrs.update(nobs_attrs())
    o_npara.attrs.update(npara_attrs())
    
    return o_dsneq

