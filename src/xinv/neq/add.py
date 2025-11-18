## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import xarray as xr
import numpy as np
from xinv.core.attrs import find_xinv_coords,find_neq_components, xunk_coords_attrs,xinv_tp,xinv_st

from xinv.core.logging import xinvlogger

from xinv.core.tools import find_ilocs
from xinv.core.grouping import find_group_coords,build_group_coord

def neqadd(dsneq:xr.Dataset, dsneqother:xr.Dataset):
    """ merge two normal equation systems"""
    try:
        N1,rhs1,x01,ltpl1,sigma01,nobs1,npara1=find_neq_components(dsneq)
        N2,rhs2,x02,ltpl2,sigma02,nobs2,npara2=find_neq_components(dsneqother)
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
    group_id_co,group_seq_co,_=find_group_coords(dsneq)

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
    Nout,rhsout,x0out,ltplout,sigma0out,nobsout,nparaout=find_neq_components(dsneq_merged)
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
    


# def add(dsneq:xr.Dataset, dsneqother:xr.Dataset):
    # """ merge two normal equation systems"""
    # try:
        # N1,rhs1,x01,ltpl1,sigma01,nobs1,npara1=find_neq_components(dsneq)
        # N2,rhs2,x02,ltpl2,sigma02,nobs2,npara2=find_neq_components(dsneqother)
    # except KeyError:
        # raise RuntimeError("Cannot find all NEQ components, are the appropriate xinv_ attributes set?")
    
    # if N1.name != N2.name or rhs1.name != rhs2.name or ltpl1.name != ltpl2.name or sigma01.name != sigma02.name or nobs1.name != nobs2.name or npara1.name != npara2.name:
        # xinvlogger.warning("Normal system variables have inconsistent variable names, using names from the first system")

    # #check for uniform sigma scaling and raise an error if not
    # sigma_ratio=sigma01/sigma02
    # if len(sigma_ratio.shape) == 0:
        # sigma_ratio_single=sigma_ratio.item()
        # hasauxdim=False 
    # else:
        # sigma_ratio_single=sigma_ratio[0].item()
        # if not (sigma_ratio == sigma_ratio_single).all():
            # raise ValueError(f"Merging NEQs with a common normal matrix but with varying sigma0 ratios over the auxiliary dimensions is not possible")
        # hasauxdim=True


    # var2_ratio=sigma_ratio_single**2
    # #check matrix compatibility
    # #check if N is in upper triangular or lower triangular form
    # if N1.attrs['xinv_state']!=N2.attrs['xinv_state']:
        # raise RuntimeError(f"Can't currently merge NEQs with different xinv_state:{N1.attrs['xinv_state']} vs {N2.attrs['xinv_state']}")
    
    # unkdim1=N1.dims[0]
    # unkdim2=N2.dims[0]
    
    # #find out the unique unknown parameters
    # # unique_unk_coord=pd.MultiIndex.from_tuples(np.unique(np.concatenate([N1[unkdim1].data,N2[unkdim2]])))
    # unique_unk_coord=xr.DataArray(np.unique(np.concatenate([N1[unkdim1].data,N2[unkdim2]])),dims=[unkdim1],name=unkdim1)
    # unique_unk_coord.attrs.update(xunk_coords_attrs(state=xinv_st.linked))

    # #find the auxiliary dimensions (ignore the unknown parameter dimension, and group_id/and seq)
    
    # xinvcoords=find_xinv_coords(dsneq,exclude=[xinv_tp.grp_id_co,xinv_tp.grp_seq_co])
    # group_id_co,group_seq_co=find_group_coords(dsneq)

    # #check if the group id and sequence coordinates are present in the first system
    # if group_id_co is not None and group_seq_co is not None:
        # #turn the unknown coordinate into a multiindex
        # grp_co=build_group_coord(unique_unk_coord,dim=unkdim1,group_id_name=group_id_co.name,group_seq_name=group_seq_co.name)
        # xinvcoords[unkdim1]=grp_co[unkdim1]
        # xinvcoords[group_id_co.name]=grp_co[group_id_co.name]
        # xinvcoords[group_seq_co.name]=grp_co[group_seq_co.name]
    # else:

        # #replace the unknow coordinate with the union version
        # xinvcoords[unkdim1]=unique_unk_coord
    # #add the proper attributes

    
    
    
    # xinvcoordsother=find_xinv_coords(dsneqother,exclude=[xinv_tp.grp_id_co,xinv_tp.grp_seq_co])
    # #possibly update with complementary coordinates from the second system
    # xinvcoords.update({ky:coord for ky,coord in xinvcoordsother.items() if ky not in xinvcoords.keys()})
    # #allocate space for the combined output normal equation system and use system one as the base template
    # dsneq_merged=xr.Dataset.xi.neqzeros(rhsdims=rhs1.dims,coords=xinvcoords)
    # #for some reason the multindex coordinate attributes do not get properly propagated
    # #so make sure they are added
    # for key,coord in xinvcoords.items():
        # if not dsneq_merged[key].attrs:
            # dsneq_merged[key].attrs.update(coord.attrs)

    # #rename the variables to the same name as in the first system
    # renamedict=dict(N=N1.name,rhs=rhs1.name,ltpl=ltpl1.name,sigma0=sigma01.name,nobs=nobs1.name,npara=npara1.name)
    # dsneq_merged=dsneq_merged.rename(renamedict)
    
    # #get links to the combined normal equation system
    # Nout,rhsout,x0out,ltplout,sigma0out,nobsout,nparaout=find_neq_components(dsneq_merged)
    # #Note: we need to be very careful when assigning values to the above views so that we insert data at the existing memory locations (!!), rather than replace the views alltogether
    # #add stuff from the first normal equation system
    
    # idx1=find_ilocs(dsneq_merged,unkdim1,N1[unkdim1].data)
    # Nout[idx1,idx1]=N1.data
    # rhsout[{unkdim1:idx1}]=rhs1.data
    
    # #note use [()] to replace the value in place
    # ltplout[()]=ltpl1
    # sigma0out[()]=sigma01
    # nobsout[()]=nobs1
    # nparaout[()]=np.int64(dsneq_merged.sizes[unkdim1])
    # #possibly add reduced parameters from the first system
    # nreduced=npara1-dsneq.sizes[unkdim1]
    # nparaout+=nreduced.astype(np.int64)


    
    # #add stuff from the second normal equation system
    # idx2=find_ilocs(dsneq_merged,unkdim2,N2[unkdim2].data)

    
    
    # #Note that we need to take care of possible different sigma0's
    # if var2_ratio != 1.0:
        # xinvlogger.info(f"Applying a rescaling of {var2_ratio} to the second normal equation system to compensate for the different apriori variances")
    
    # Nout[idx2,idx2]+=var2_ratio*N2.data
    
    # rhsout[{unkdim1:idx2}]+=var2_ratio*rhs2.data
    # ltplout+=var2_ratio*ltpl2
    # #note dsneq_merged['sigma0'] stays the same 
    # nobsout+=nobs2
    # #possibly add the amount of reduced parameters from the second system
    # nreduced=npara2-dsneqother.sizes[unkdim2]
    # nparaout+=nreduced.astype(np.int64)

    
    # return dsneq_merged                     
    


