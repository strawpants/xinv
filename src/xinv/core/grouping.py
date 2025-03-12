## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl
import numpy as np
import xarray as xr
from xinv.core.attrs import group_coord_attrs

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

    if stack_dim is not None:
        dsout=dsout.stack({stack_dim:[group_id_dim,group_seq_dim]})
    
    return dsout


