## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl


from xinv.core.attrs import ltpl_attrs,sigma0_attrs,N_attrs,rhs_attrs,nobs_attrs,npara_attrs,xinv_tp,x0_attrs
from xinv.core.logging import xinvlogger

import numpy as np
import xarray as xr

def zeros(rhsdims,coords,lower=0,norder='C'):
        
        
        #figure out the shape of the rhs from the provided coordinates
        # rhsshape=[len(coords[dim]) for dim in rhsdims]
        rhsshape=[]
        grpco=None
        for i,dim in enumerate(rhsdims):
            if dim not in coords:
                raise KeyError(f"Coordinate {dim} not found in the provided coordinates")
            if type(coords[dim]) == xr.core.coordinates.Coordinates:
                #Extract correct subcoordinate from the multindex group coordinate
                co=coords[dim][dim]
                grpco=coords[dim]
            else:
                co=coords[dim]
            #Note: We assume that the first dimension is the unknown parameter dimension
            if i== 0 and not "xinv_type" in co.attrs:
<<<<<<< HEAD
                xinvlogger.warning(f"Unknown parameter dimension is assumed to be the first one {rhsdims[0]}")
            elif co.attrs["xinv_type"] != xinv_tp.unk_co:
                raise ValueError("The unknown parameter does not correspond to the first dimenion of the right hand side")
=======
                xinvlogger.warning("Unknown parameter dimension is assumed to be the first one {rhsdims[0]}")
                if co.attrs["xinv_type"] != xinv_tp.unk_co:
                    raise ValueError("The unknown parameter does not correspond to the first dimenion of the right hand side")
>>>>>>> 66694e0 (Added transform function, updated fwdoperator interfacing, and more)
            rhsshape.append(len(co))


        #allocate space
        N=([rhsdims[0],rhsdims[0]+'_'],np.zeros([rhsshape[0],rhsshape[0]],order=norder))
        rhs=(rhsdims,np.zeros(rhsshape))
        x0=(rhsdims,np.zeros(rhsshape))

        ltpl=(rhsdims[1:],np.zeros(rhsshape[1:]))
        sigma0=(rhsdims[1:],np.zeros(rhsshape[1:]))
        nobs=(rhsdims[1:],np.zeros(rhsshape[1:],dtype=np.int64))
        npara=(rhsdims[1:],np.zeros(rhsshape[1:],dtype=np.int64))

        if grpco is not None:
            dsneq=xr.Dataset(dict(N=N,rhs=rhs,x0=x0,ltpl=ltpl,nobs=nobs,npara=npara,sigma0=sigma0),coords=grpco).assign_coords({ky:co for ky,co in coords.items() if ky != rhsdims[0]})
        else:
            dsneq=xr.Dataset(dict(N=N,rhs=rhs,x0=x0,ltpl=ltpl,nobs=nobs,npara=npara,sigma0=sigma0),coords=coords)
        #add attributes
        dsneq.N.attrs.update(N_attrs(lower))
        dsneq.rhs.attrs.update(rhs_attrs())
        dsneq.x0.attrs.update(x0_attrs())
        dsneq.ltpl.attrs.update(ltpl_attrs('apriori'))
        dsneq.sigma0.attrs.update(sigma0_attrs('apriori'))
        dsneq.nobs.attrs.update(nobs_attrs())
        dsneq.npara.attrs.update(npara_attrs())

        return dsneq


    
def reduce(dsneq:xr.Dataset, idx):
    #tbd reduce (implicitly solve) variables from a normal equation system spanned by idx
    raise NotImplementedError("Reduce operation not yet implemented")

def fix(dsneq:xr.Dataset, idx):
    #tbd fix and (remove) solve) variables from a normal equation system spanned by idx
    raise NotImplementedError("Fix operation not yet implemented")

<<<<<<< HEAD
def set_apriori(dsneq:xr.Dataset, daapri:xr.DataArray):
    #tbd set/change apriori values in a normal equation system
    raise NotImplementedError("Set apriori values not yet implemented")


=======
    unkdim=N.dims[0]
    mask=~dsneq[unkdim].isin(idx).data


    N_kept=N.loc[{unkdim:mask,unkdim+"_":mask}]
    rhs_kept=rhs.loc[{unkdim:mask}]

    new_unk=dsneq[unkdim].loc[{unkdim:mask}]
    new_unk.attrs.update(xunk_coords_attrs(state=xinv_st.linked)) ## check the state

    coords={unkdim:new_unk}

    nfix=len(idx)
    npara=npara-nfix
    npara=xr.DataArray(npara,attrs=npara_attrs())
    npara.attrs.update(npara.attrs)

    # compute posteriori sigma0
    sigma0=xr.DataArray(np.sqrt(ltpl/(nobs-npara)),name="sigma0")
    sigma0.attrs.update(sigma0_attrs('posteriori'))

    dsneq_kept=xr.Dataset.xi.neqzeros(rhsdims=rhs_kept.dims,coords=coords)
    renamedict=dict(N=N.name,rhs=rhs.name,ltpl=ltpl.name,sigma0=sigma0.name,nobs=nobs.name,npara=npara.name)    
    dsneq_kept=dsneq_kept.rename(renamedict)


    dsneq_kept["N"]=N_kept
    dsneq_kept.N.attrs.update(N_attrs(lower=lower))
    
    dsneq_kept["rhs"]=rhs_kept
    dsneq_kept.rhs.attrs.update(rhs.attrs)

    dsneq_kept["ltpl"]=ltpl
    dsneq_kept.ltpl.attrs.update(ltpl.attrs)

    dsneq_kept["sigma0"]=sigma0

    dsneq_kept["nobs"]=nobs
    dsneq_kept.nobs.attrs.update(nobs.attrs)

    dsneq_kept["npara"]=npara

    return dsneq_kept
    

def set_apriori(dsneq:xr.Dataset, daapri:xr.DataArray, absolute=False, inplace=False):
    
    """ Change apriori values in a normal equation system """

    if not inplace:
    #copy entire NEQ and operate on that one in place
        dsneq=dsneq.copy(deep=True)

    N,rhs,ltpl,_,_,_=find_neq_components(dsneq)

    try:
        x0=find_component(dsneq,xinv_tp.x0)
    except:
        # it may not be there
        x0=xr.zeros_like(rhs).assign_attrs(x0_attrs())

    
    if absolute:
        ## take the differences
        deltax0=daapri-x0
        x0_new=daapri
        x0[()]=x0_new

    else:
        # compute the deltax0 internally
        deltax0=daapri
        x0_new=x0+deltax0        
        x0[()]=x0_new


    ## update ltpl using the a priori in place
    ltpl[()]+= (-2*deltax0.transpose().data@rhs.data) + (deltax0.transpose().data@N.data@deltax0.data)

    ## update rhs using the a priori in place
    rhs[()]-=N.data.dot(deltax0.data)
    
    
    if inplace:
        return None
    else:
        return dsneq
    
>>>>>>> 66694e0 (Added transform function, updated fwdoperator interfacing, and more)

    


    
    
