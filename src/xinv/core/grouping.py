## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl
import numpy as np
import xarray as xr
from xinv.core.attrs import find_component, get_xunk_size_coname, group_id_attrs, group_seq_attrs,find_xinv_coords, xunk_coords_attrs,xinv_tp,xinv_st,unlink,is_linked
import pandas as pd
from xinv.core.logging import xinvlogger
from xinv.core.tools import find_ilocs


def find_group_coords(dsneq,grpdim=None,assoc_coords=None):
    """
    Find the group id and sequence coordinates in a dataset
    Parameters
    ----------
    dsneq : xr.Dataset
        Contains Normal equations system elements or a solution thereof
    grpdim: str
        name of the dimension which matches the group coordinates
    Returns
        
    -------
    (group_id_co,group_seq_co,group_assoc_co) : xr.DataArray
        The group id and seq coordinate or (None,None,None) if not found
    
    """
    group_id_co=None
    group_seq_co=None
    group_asso_co=None 
    #try to find heuristically by naming
    if grpdim is not None:
        for k,var in dsneq.variables.items():
            if len(var.dims) > 0 and var.dims[0] == grpdim:
                if k.endswith('grp_id'):
                    group_id_co=dsneq[k]
                elif k.endswith('grp_seq'):
                    group_seq_co=dsneq[k]
    else:
        try:
            group_id_co=find_component(dsneq,xinv_tp.grp_id_co)
            group_seq_co=find_component(dsneq,xinv_tp.grp_seq_co)
        except KeyError:
            pass
    #check if they are currently linked
    if group_seq_co is not None and group_id_co is not None:
        if not( is_linked(group_seq_co) and is_linked(group_id_co)):
            #reset
            group_seq_co=None
            group_id_co=None
            
    #try to find the associated coordinates found in the group_id
    if group_id_co is not None:
        group_asso_co={}
        for asso_co_name in np.unique(group_id_co):

            if asso_co_name in dsneq:
                group_asso_co[asso_co_name]=dsneq[asso_co_name]
            elif assoc_coords is not None and asso_co_name in assoc_coords:
                group_asso_co[asso_co_name]=assoc_coords[asso_co_name]
            else:
                xinvlogger.warning(f"Missing associated group id coordinate: {asso_co_name}, consider adding from original source")
                group_asso_co[asso_co_name]=None


        
    return group_id_co,group_seq_co,group_asso_co

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
    xinvlogger.warning("expand-as_group is deprecated, use as_group instead")

    dsout=dsin.expand_dims(dim=group_id_dim,axis=None).rename({group_dim:group_seq_dim}).assign_coords({group_id_dim:(group_id_dim,[group_dim]),group_seq_dim:(group_seq_dim,np.arange(len(dsin[group_dim])))})
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
        
        #possibly rename dangling transpose dimensions
        dimrename={}
        for dim in dsout.dims:
            if dim == group_dim+'_' and dim not in dsout.coords:
                dimrename[dim]=stack_dim+"_"
        if dimrename:
            dsout=dsout.rename_dims(dimrename)
        

        #possibly fix the order of some matrices which may now be transposed ue to the stacking
    return dsout

def as_group(dsin,arg=None,*,group_id_dim="xinv_grp_id",group_seq_dim="xinv_grp_seq",lookup_co=None,**kwargs):
    
    #extract mapping either from arg or from kwargs arguments
    if arg is not None:
        old_coord_name,group_coord_name=next(iter(arg.items()))
    elif len(kwargs) == 1:
        old_coord_name,group_coord_name=next(iter(kwargs.items()))
    else:
        raise ValueError("No group mapping provided. use either named arguments oldcoord=new_group_coord, or a provide a dictionary with the mapping")

    #create a  new Dataset using all of the existing coordinates from the input
    dsout=xr.Dataset(coords=dsin.coords,attrs=dsin.attrs)
    
    # unlink old coordinate
    unlink(dsout[old_coord_name])
    if lookup_co is not None:
        try:
            grp_idx=find_ilocs(lookup_co,old_coord_name,dsin[old_coord_name])
            #overwrite old coordinate with the lookup one for consistency
            dsout=dsout.assign_coords({old_coord_name:lookup_co[old_coord_name]})
        except:
            raise RuntimeError("Can not find all of parameters in the lookup dataset")
    else:
        grp_idx=np.arange(dsin.sizes[old_coord_name])
        #just incrementally increasing sequence
    mi=pd.MultiIndex.from_tuples([(old_coord_name,i) for i in grp_idx],names=[group_id_dim,group_seq_dim])

    groupcoord=xr.Coordinates.from_pandas_multiindex(mi,dim=group_coord_name)
    
    dsout=dsout.assign_coords(groupcoord)
    
    # add xinv attributes 

    dsout[group_coord_name].attrs.update(xunk_coords_attrs(state="linked"))    
    dsout[group_id_dim].attrs.update(group_id_attrs(state="linked"))
    dsout[group_seq_dim].attrs.update(group_seq_attrs(state="linked"))

    #copy and rename relevant variables
    for vname in dsin.variables:
        if vname not in dsout.variables:
            #only copy stuff which is not yet in there
            newdims=[dim.replace(old_coord_name,group_coord_name) for dim in dsin[vname].dims]
            dsout[vname]=(newdims,dsin[vname].data,dsin[vname].attrs) 


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
    try:
        if type(dsneq.indexes[groupname]) == pd.MultiIndex:
            groupismindex=True
        else:
            groupismindex=False
    except: 
        groupismindex=False

    #create an index vector of the data asssociated with the groupname
    
    #find the group id and sequence dimensions
    group_id_name=find_component(dsneq,xinv_tp.grp_id_co).name
    group_seq_name=find_component(dsneq,xinv_tp.grp_seq_co).name
    
    
    grpidx=dsneq[group_id_name] == groupname
    
    try:
        unkdim=find_component(dsneq,xinv_tp.rhs).dims[0]
    except KeyError:
        unkdim=find_component(dsneq,xinv_tp.solest).dims[0]
    unkdim_=f"{unkdim}_"

    try:
        try:
            find_component(dsneq,xinv_tp.N)
        except KeyError:
            #ok so try find a covariance matrix
            find_component(dsneq,xinv_tp.COV)
        #extract the relevant subsections
        dsout=dsneq[{unkdim:grpidx.data,unkdim_:grpidx.data}] 
        rename={unkdim:groupname,unkdim_:groupname+"_"}
        hasmatrix=True
    except KeyError:
        #neither N or COV was found
        dsout=dsneq[{unkdim:grpidx.data}] 
        rename={unkdim:groupname}
        hasmatrix=False

    
    groupcoord=dsout[groupname][dsout[group_seq_name]]
    
    #multindex and all levels must be dropped
    dropvars=[nm for nm in dsout.get_index(unkdim).names]
    dropvars.append(unkdim)
    if groupismindex:
        #also add all levels of the group to drop
        dropvars.extend([nm for nm in dsout.get_index(groupname).names])

    #also drop the groupname varibale (will be readded later)
    dropvars.append(groupname)
    dsout=dsout.drop_vars(dropvars).rename(rename)
    #extract the relevant original coordinate components
    if groupismindex:
        groupcoord=pd.MultiIndex.from_tuples(groupcoord.data,names=dsneq.get_index(groupname).names)
        groupcoord=xr.Coordinates.from_pandas_multiindex(groupcoord,dim=groupname)
        dsout=dsout.assign_coords(groupcoord)
    else:

        dsout=dsout.assign_coords({groupname:(groupname,groupcoord.data)})
    #possibly rebuild the multindex if the original group was a multindex
    #add xinv attributes
    dsout[groupname].attrs.update(xunk_coords_attrs(state=xinv_st.linked))

    return dsout

def reindex_groups(dsneq,group_dim=None,assoc_coords=None):
    """
    Rebuilds the groups,sequences into a multiIndex of a dataset containing a Normal equation system or solution thereof (e.g. read from a file)
    Parameters
    ----------
    dsneq : xr.Dataset
        Contains Normal equations system elements or a solution thereof
    group_dim:str
        name of a specific group_dimension to look for group coordinates
    assoc_coords: dict
        dictionary with auxiliary source coordinates which can be copied when they are associated with the group_id's
    returns: xr.Dataset
        An xarray.Dataset with a valid multindex

    """
    
    

    
#    try:
 #       NorCOV=find_component(dsneq,xinv_tp.N)
 #   except KeyError:
 #       NorCOV=find_component(dsneq,xinv_tp.COV)
    
 #   unkdim=NorCOV.dims[0]
    if group_dim is None:
        try:
            _,group_dim=get_xunk_size_coname(dsneq)
        except StopIteration:
            #try in a different way
                
            try:
                NorCOV=find_component(dsneq,xinv_tp.N)
            except KeyError:
                NorCOV=find_component(dsneq,xinv_tp.COV)
    
            group_dim=NorCOV.dims[0]
 
    #try to find a group id and sequence dimensions
    group_id_co,group_seq_co,asso_co=find_group_coords(dsneq,group_dim,assoc_coords)
        
    if group_id_co is None or group_seq_co is None:
        raise RuntimeError("Group id and sequence coordinates can not be found from xinv attributes")
    

    #recreate the multindex based on the group and seq id
    grp_co=build_group_coord([dsneq[group_id_co.name].data,dsneq[group_seq_co.name].data],dim=group_dim,group_id_name=group_id_co.name,group_seq_name=group_seq_co.name)
    
    #possibly augment with associated coordinates

    #import pdb; pdb.set_trace()
    grp_co.update(asso_co)

    dsneq=dsneq.drop_vars([group_id_co.name,group_seq_co.name]).assign_coords(grp_co)

    return dsneq


def rename_groups(dsneq,grpmap):
    #try to find a group id and sequence dimensions
    group_id_co,group_seq_co,_=find_group_coords(dsneq)
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


def split_as_groups(dsneq,group_ids:xr.DataArray,stack_dim=None):
    """
    """
    
    unksz,unkdim=get_xunk_size_coname(dsneq)

    if unksz != len(group_ids):
        raise ValueError(f"Size of group_ids {len(group_ids)} does not match the size of the unknown dimension {unksz} in the dataset")

    #make new groupcoordinates
    grpseqtrack={grp:0 for grp in np.unique(group_ids.data)}
    splitcoords={grp:[] for grp in grpseqtrack.keys()}
    grpdata=[]
    for idx,group_id in enumerate(group_ids.data):
        iseq=grpseqtrack[group_id]
        grpdata.append((group_id,iseq))  # sequence number is always 0 for now
        #copy original coordinate value
        splitcoords[group_id].append(dsneq[unkdim].data[idx])
        grpseqtrack[group_id] += 1
    
    if stack_dim is None:
        stack_dim= f"{unkdim}_spl"
    groupco= build_group_coord(grpdata,dim=stack_dim)
    
    #setup new coordinates
    attrs=xunk_coords_attrs(state=xinv_st.unlinked)
    newcoords={grp:(grp,co,attrs) for grp,co in splitcoords.items()}
    
    dsneq=dsneq.drop_vars([unkdim]).rename({unkdim:stack_dim,f"{unkdim}_":f"{stack_dim}_"}).assign_coords(newcoords).assign_coords(groupco)

    # breakpoint()

    return dsneq
