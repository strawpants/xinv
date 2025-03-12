## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl
import numpy as np
import xarray as xr
from xinv.core.attrs import find_component, group_coord_attrs, group_id_attrs

def expand_as_group(dsin,group_dim,group_id_dim="xinv_grp_id",group_seq_dim="xinv_grp_seq",stack_dim=None):
    """
    Expand a DataArray or Dataset along a group dimension, adding a group_id_dim and group_seq_dim to the dataset
    Parameters
    ----------
    dsin : xr.DataArray or xr.Dataset
        The input data to expand
        
    group_dim : str
        The dimension in the original data to expand
        
    group_id_dim : str, optional
        The name of the new dimension to hold the group id
        
    group_seq_dim : str, optional
        The name of the new dimension to hold the sequence number within the group
        
    stack_dim : str, optional
        The name of the new dimension to stack the group_id_dim and group_seq_dim
        

    Returns
    -------
    xr.Dataset with the expanded coordinates
        

    """
    dsout=dsin.expand_dims(dim=group_id_dim).rename({group_dim:group_seq_dim}).assign_coords({group_id_dim:(group_id_dim,[group_dim]),group_seq_dim:(group_seq_dim,np.arange(len(dsin[group_dim])))})
    if type(dsout) == xr.DataArray:
        dsout=dsout.to_dataset()
    # dsout=dsout.rename({group_dim:group_seq_dim})

    #For retrieval purposes later: copy the original parameters back to the old dimension
    dsout=dsout.assign_coords({group_dim:dsin[group_dim]})
    
    # add xinv attributes to mark the original coordinates as unlinked
    dsout[group_dim].attrs.update(group_coord_attrs(state="unlinked"))    

    dsout[group_id_dim].attrs.update(group_id_attrs(state="linked"))

    if stack_dim is not None:
        dsout=dsout.stack({stack_dim:[group_id_dim,group_seq_dim]})
    
    return dsout



def get_group(dsneq,groupname,group_id_dim="xinv_grp_id",group_seq_dim="xinv_grp_seq"):
    """
    Retrieve a parameter group from a Dataset which contains a Normal equation system or solution thereof
    Parameters
    ----------
    dsneq : xr.Dataset
        Contains Normal equations system elements or a solution thereof
        
    groupname : str
        Groupname to retrieve
    
    returns: xr.Dataset
        The subset of the input which is valid for the groupname

    """
    
    #create an index vector of the data asssociated with the groupname
    grpidx=dsneq[group_id_dim] == groupname
    
    try:
        unkdims=find_component(dsneq,"N").dims
    except KeyError:
        unkdims=find_component(dsneq,"COV").dims

    #extract the relevant subsections
    dsout=dsneq[{unkdims[0]:grpidx.data,unkdims[1]:grpidx.data}] 
    #extract the relevant original coordinate components
    groupcoord=dsout[groupname][dsout[group_seq_dim]]
    #multindex and all levels must be dropped
    dropvars=[nm for nm in dsout.get_index(unkdims[0]).names]
    dropvars.append(unkdims[0])
    #also drop the groupname varibale (will be readded later)
    dropvars.append(groupname)

    dsout=dsout.drop_vars(dropvars).rename({unkdims[0]:groupname,unkdims[1]:groupname+"_"}).assign_coords({groupname:(groupname,groupcoord.data)})
    dsout[groupname].attrs.update(group_coord_attrs(state="linked"))

    return dsout
