## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import numpy as np
import xarray as xr


def fix(dsneq:xr.Dataset, idx):
    
    """fix and remove solved variables from a normal equation system spanned by idx"""
    
    N,rhs,ltpl,sigma0,nobs,npara=find_neq_components(dsneq)
    
    if N.attrs['xinv_state'] == xinv_st.symU:
        lower=0
    elif N.attrs['xinv_state'] == xinv_st.symL:
        lower=1
    else:
        raise RuntimeError(f"Don't know (yet) how to cope with normal matrix state:{N.attrs['xinv_state']}")

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