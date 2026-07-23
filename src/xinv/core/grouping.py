## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl
import numpy as np
import xarray as xr
from xinv.core.attrs import find_component, get_xunk_size_coname, group_id_attrs, group_seq_attrs,find_xinv_coords, xunk_coords_attrs,xinv_tp,xinv_st,unlink,is_linked
import pandas as pd
from xinv.core.logging import xinvlogger,deprecated
from xinv.core.tools import find_ilocs,find_unk_idxv2,select
import re

@deprecated("testing custom deprecation warning")
def test_depr():
    print("hi")

def build_group_index(grpcoords,name='xinv_unk'):
    """
        Build a Pandas MultiIndex by using the input coordinates as independent levels, whilst 
    """
        
    #create levels
    levels=[]
    level_len=[]
    names=[]
    
    for co in grpcoords:
        levels.append(np.append(co.values,None))
        level_len.append(co.size)
        names.append(co.name)
    nlev=len(level_len)
    co_ln=np.sum(level_len)
    
    codes=np.full([nlev,co_ln],-1)
    #plugin the slices for the appropriate levels
    shft=0
    for i,lev in enumerate(level_len):
        codes[i,shft:shft+lev]=np.arange(lev)
        shft+=lev
    
    return pd.MultiIndex(levels=levels,codes=codes,names=names)
    
def deserialize_groups(dsneq):
    """
    
        rebuilds the group based multindex from the levels and data in a serialized dataset
    """
    
    varnames=[var for var,val in dsneq.data_vars.items()] 
    #also look in the coords
    varnames.extend([var for var,val in dsneq.coords.items()] )
    
    #find a potential multilevel variable which needs to be deserialized
    tmp=[name for name in varnames if name.endswith("_mc")]
    if len(tmp) != 1:
        raise ValueError("None or Too many deserialization candidates (ending with '_mc') found")



    unkdim=tmp[0][:-3]
    groupnames=dsneq['group_mn'].data
    ngroups=groupnames.size

    #figure out level data
    levels=[]
    for group in groupnames:
        lnames=[var for var in varnames if re.fullmatch(f'({group}_ml)|({group}_mlt[0-9])',var)]
        if len(lnames) == 1:
            #single coordinate data only
            levels.append(dsneq[lnames[0]].data)
            # levels.append(np.append(dsneq[lnames[0]].data,None))
        else:
            lnames=sorted(lnames) #make sure to sort lexigraphically)
            ntlev=len(lnames)
            args=[dsneq[name].data.tolist() for name in lnames]
            level=[None if np.all(np.isin(tp,['',np.nan])) else tp for tp in zip(*args)]
            # level=[tp for tp in zip(*args)]
            # level.append(None)
            levels.append(level)

    
    midx=xr.Coordinates.from_pandas_multiindex(pd.MultiIndex(levels=levels,codes=dsneq[unkdim+"_mc"].data,names=groupnames),unkdim)
    drop_vars=[var for var in varnames if re.match(r'.*_m[cnl]t?[0-9]?$',var)]
    dsout=dsneq.drop_vars(drop_vars).assign_coords(midx)
    dsout[unkdim].attrs.update(xunk_coords_attrs(state=xinv_st.linked))
    return dsout

def serialize_groups(dsneq):
    """
        Serialize (nested) multindices associated with the unknown coordinate index so they can be written to a file

    Parameters
    ----------
    dsneq : 
        

    Returns
    -------
    
        
    
        

    """
    
    unk_dim,_=dsneq.xi.unknown_dim()
    midx=dsneq.get_index(unk_dim)
    
    if not hasattr(midx,'levels'):
        #nothing to do
        return dsneq
    
    #retrieve levels and add to dataset
    for name,level in zip(midx.names,midx.levels):
        sername=name+"_ml"
        #possibly split up tuplesi with mixed entries in different variables
        if type(level[0]) == tuple:
            ntlev=len(level[0])
            for i in range(ntlev):
                s_sername=sername+f"t{i}"
                # dsneq[s_sername]=(sername,[tp[i] for tp in level[0:-1]])
                dsneq[s_sername]=(sername,[tp if tp is None else tp[i] for tp in level])
        else:
            # no need to further serialize this
            dsneq[sername]=(sername,level)
    #also add the codes so we can reconstruct the multiindex later
    dsneq[unk_dim+"_mc"]=(["ngroups",unk_dim],midx.codes)
    dsneq["group_mn"]=(["ngroups"],midx.names) 
    drop_vars=[nm for nm in midx.names]
    drop_vars.append(unk_dim)
    
    dsneq=dsneq.drop_vars(drop_vars)
    return dsneq
    


@deprecated("You shouldn't be using old group style coordinates in this version anymore")
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

@deprecated("You shouldn't be using old group style coordinates in this version anymore")
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

@deprecated("expand_as_group is deprecated, use as_group instead")
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

# def add_level(dsin,arg=None,**kwargs):
    # """
        # Expand the index to a multiindex (or add an additional level) with a new groupnamer and value
    # """
    # #extract mapping either from arg or from kwargs arguments
    # if type(arg) is dict and len(arg) == 1:
        # group_name,group_value=next(iter(arg.items()))
    # elif len(kwargs) == 1:
        # group_name,group_value=next(iter(kwargs.items()))
    # else:
        # raise ValueError("No or ambigious group mapping provided. use either named arguments group_name=group_value, or a provide a dictionary with the mapping")

    # dsout=dsin.copy()
    
    # #get the current unknown index
    # unkdim,_=dsin.xi.unknown_dim()
    # unk_idx=dsin.get_index(unkdim)

    # dftmp = unk_idx.to_frame()

    # # Insert new level at specified location
    # dftmp.insert(0, group_name, group_value)

    # # Convert back to MultiIndex
    # dsout=dsout.assign_coords({unkdim:pd.MultiIndex.from_frame(dftmp)})
    # #set attrs
    # dsout[unkdim].attrs.update(xunk_coords_attrs(state="linked"))
    # return dsout

def get_group(dsneq,level_name):
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
    dsout=select(dsneq,**{level_name:None},inverse=True)
    newidx=dsout[level_name].data
    unkdim,unkdim_=dsneq.xi.unknown_dim()
    # dsout=dsout.reset_index(unkdim).drop_vars([name for name in self.index.names if name != level_name]).rename({unkdim:level_name}).set_xindex(level_name)
    dsout=dsout.reset_index(unkdim,drop=True).rename_dims({unkdim:level_name}).assign_coords({level_name:newidx})

    if unkdim_ is not None:
        dsout=dsout.rename_dims({unkdim_:level_name+"_"})
    idx=dsout.get_index(level_name)
    levnamekey=level_name+"_levels"
    if levnamekey in dsout and type(idx) == pd.Index:
        #build sub multiindex
        midx=pd.MultiIndex.from_tuples(idx,names=dsout[levnamekey].data)
        dsout=dsout.reset_index(level_name,drop=True).drop_vars(levnamekey).assign_coords(xr.Coordinates.from_pandas_multiindex(midx,level_name))
    #reassgin attributes
    dsout[level_name].attrs.update(xunk_coords_attrs(state=xinv_st.linked))
    return dsout

@deprecated("reindex_groups uses outdated group coordinate structures")
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


def rename_groups(dsneq,grpmap=None,**kwargs):
    if grpmap is not None and len(kwargs) != 0:
        raise ValueError("Either supply a mapping as a dictionary of named arguments according to the levels to be renamed")
    
    if len(kwargs) >=1:
        grpmap=kwargs

    unkdim,_=dsneq.xi.unknown_dim()
    xinv_idx=dsneq.get_index(unkdim)
    xinv_idx=xinv_idx.set_names(grpmap)
    dsneq=dsneq.reset_index(unkdim,drop=True).assign_coords(xr.Coordinates.from_pandas_multiindex(xinv_idx,unkdim))
    dsneq[unkdim].attrs.update(xunk_coords_attrs(state=xinv_st.linked))
    return dsneq


@deprecated("You shouldn't be using old group style coordinates in this version anymore")
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
