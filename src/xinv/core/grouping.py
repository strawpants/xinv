## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl
import numpy as np
import xarray as xr
from xinv.core.attrs import find_component, group_id_attrs, group_seq_attrs,find_xinv_coords, xunk_coords_attrs,xinv_tp,xinv_st
import pandas as pd
from xinv.core.logging import xinvlogger

def find_group_coords(dsneq):
    """
    Find the group id and sequence coordinates in a dataset
    Parameters
    ----------
    dsneq : xr.Dataset
        Contains Normal equations system elements or a solution thereof
    Returns
        
    -------
    (group_id_co,group_seq_co) : xr.DataArray
        The group id and seq coordinate or (None,None) if not found
    
    """

    try:
        group_id_co=find_component(dsneq,xinv_tp.grp_id_co)
        group_seq_co=find_component(dsneq,xinv_tp.grp_seq_co)
    except KeyError:
        group_id_co=None
        group_seq_co=None
    
    return group_id_co,group_seq_co

def build_group_coord(data,dim='xinv_unk',group_id_name="xinv_grp_id",group_seq_name="xinv_grp_seq"):
   
    if type(data) == xr.DataArray:
        data=data.data
    if type(data[0]) == tuple:
        mi=pd.MultiIndex.from_tuples(data, names=[group_id_name,group_seq_name])
    else:
        mi=pd.MultiIndex.from_arrays(data, names=[group_id_name,group_seq_name])

    grpcoord=xr.Coordinates.from_pandas_multiindex(mi,dim=dim)
    #add the proper xinv attributes
    grpcoord[dim].attrs.update(xunk_coords_attrs(state=xinv_st.linked))
    #note we need to re-add the group id and seq coordinate attributes
    grpcoord[group_id_name].attrs.update(group_id_attrs(state=xinv_st.linked))
    grpcoord[group_seq_name].attrs.update(group_seq_attrs(state=xinv_st.linked))
    return grpcoord

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
    dsout[group_dim].attrs.update(xunk_coords_attrs(state="unlinked"))    
    dsout[group_id_dim].attrs.update(group_id_attrs(state="linked"))
    dsout[group_seq_dim].attrs.update(group_seq_attrs(state="linked"))

    if stack_dim is not None:
        dsout=dsout.stack({stack_dim:[group_id_dim,group_seq_dim]})
        dsout[stack_dim].attrs.update(xunk_coords_attrs(state="linked"))
    
    return dsout



def get_group(dsneq,groupname):
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
    xinvcoords=find_xinv_coords(dsneq)

    #create an index vector of the data asssociated with the groupname
    
    #find the group id and sequence dimensions
    group_id_name=find_component(dsneq,xinv_tp.grp_id_co).name
    group_seq_name=find_component(dsneq,xinv_tp.grp_seq_co).name
    
    
    grpidx=dsneq[group_id_name] == groupname
    
    try:
        unkdims=find_component(dsneq,xinv_tp.N).dims
    except KeyError:
        unkdims=find_component(dsneq,xinv_tp.COV).dims

    #extract the relevant subsections
    dsout=dsneq[{unkdims[0]:grpidx.data,unkdims[1]:grpidx.data}] 
    #extract the relevant original coordinate components
    groupcoord=dsout[groupname][dsout[group_seq_name]]
    #multindex and all levels must be dropped
    dropvars=[nm for nm in dsout.get_index(unkdims[0]).names]
    dropvars.append(unkdims[0])
    #also drop the groupname varibale (will be readded later)
    dropvars.append(groupname)

    dsout=dsout.drop_vars(dropvars).rename({unkdims[0]:groupname,unkdims[1]:groupname+"_"}).assign_coords({groupname:(groupname,groupcoord.data)})
    dsout[groupname].attrs.update(xunk_coords_attrs(state="linked"))

    return dsout

def reindex_groups(dsneq):
    """
    Rebuilds the groups,sequences into a multiIndex of a dataset containing a Normal equation system or solution thereof (e.g. read from a file)
    Parameters
    ----------
    dsneq : xr.Dataset
        Contains Normal equations system elements or a solution thereof
    returns: xr.Dataset
        An xarray.Dataset with a valid multindex

    """
    
    

    
    try:
        NorCOV=find_component(dsneq,xinv_tp.N)
    except KeyError:
        NorCOV=find_component(dsneq,xinv_tp.COV)
    
    unkdim=NorCOV.dims[0]

    
    #try to find a group id and sequence dimensions
    group_id_co,group_seq_co=find_group_coords(dsneq)
        
    if group_id_co is None or group_seq_co is None:
        raise RuntimeError("Group id and sequence coordinates can not be found from xinv attributes")
    

    #recreate the multindex based on the group and seq id
    grp_co=build_group_coord([dsneq[group_id_co.name].data,dsneq[group_seq_co.name].data],dim=unkdim,group_id_name=group_id_co.name,group_seq_name=group_seq_co.name)
    dsneq=dsneq.drop_vars([group_id_co.name,group_seq_co.name]).assign_coords(grp_co)

    return dsneq


def rename_groups(dsneq,grpmap):
    #try to find a group id and sequence dimensions
    group_id_co,group_seq_co=find_group_coords(dsneq)
    if group_id_co is None or group_seq_co is None:
        raise RuntimeError("Group id and sequence coordinates can not be found from xinv attributes, no consistent renaming possible")
    #first remap the coordinate names themselves
    dsneq=dsneq.rename(grpmap)
    #figure out the linked unknown parameter dimensions

    xinvcoords=find_xinv_coords(dsneq,include=xinv_tp.unk_co,state=xinv_st.linked)
    unk_name=next(iter(xinvcoords.keys()))
    dsneq=dsneq.reset_index(unk_name) 
    for grp,grpnew in grpmap.items():
        dsneq[group_id_co.name]=xr.where(dsneq[group_id_co.name] == grp,grpnew,dsneq[group_id_co.name])
        #update attributes
        dsneq[group_id_co.name].attrs.update(group_id_attrs(state=xinv_st.linked))
    #rebuild the index


    return reindex_groups(dsneq)
