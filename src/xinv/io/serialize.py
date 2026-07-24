
import xarray as xr
from sparse import COO

def coo_serialize(ds):
    """
    Serializes an xarray with COO sparse matrix variables to one with separate variables. which can be writen to a file.
    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Input xarray object with sparse COO matrix variables.
    Returns
    -------
    xarray.Dataset
        Modified xarray object with separate variables for COO data and coordinates.
    """

    if type(ds) is not xr.Dataset:
        dsmod=ds.to_dataset(name='data',promote_attrs=True)
    else:
        dsmod=ds.copy()

    for vname,var in dsmod.variables.items():
        if hasattr(var.data,'format'):
            sparse_type=var.data.format
            if sparse_type == 'coo':
                #extract data and index arrays and register them as a set of new variables
                vname_co=f"{vname}_coo_co"
                vname_data=f"{vname}_coo_data"
                vname_dim=f"{vname}_coo_nnz"
                orig_dims=list(var.dims)
                attrs=var.attrs.copy()
                attrs.update(dict(coo_name=f"COO data array for variable {vname}",orig_name=vname,orig_dims=orig_dims))
                dsmod[vname_data]=([vname_dim],var.data.data,attrs)
                dsmod[vname_co]=(['coo_dim',vname_dim],var.data.coords,dict(long_name=f"COO coordinate array for variable {vname}",orig_name=vname,orig_dims=orig_dims))
                #remove the original variable
                dsmod=dsmod.drop_vars(vname)

            else:
                raise ValueError(f"Cannot currently handle sparse type {sparse_type} for variable {vname}")
    return dsmod



def coo_deserialize(ds):
    """
    Deserializes an xarray with separate COO sparse matrix variables to one with COO sparse matrix variables.
    Parameters
    ----------
    ds : xarray.Dataset
        Input xarray object with separate variables for COO data and coordinates.
    Returns
    -------
    xarray.Dataset
        Modified xarray object with sparse COO matrix variables.
    """
    dsmod=ds.copy()

    for vname,var in dsmod.variables.items():
        if vname.endswith('_coo_data'):
            #get the original variable name
            orig_name=var.attrs['orig_name']
            orig_dims=var.attrs['orig_dims']
            vname_co=f"{orig_name}_coo_co"
            vname_dim=f"{orig_name}_coo_nnz"
            shape=[dsmod.sizes[d] for d in orig_dims]
            #reconstruct sparse array
            dmat=COO.from_iter(zip(dsmod[vname_co].data.T,var.data),shape=shape)
            attrs={ky:val for ky,val in var.attrs.items() if ky not in ['orig_name','orig_dims']} 

            dsmod[orig_name]=(orig_dims,dmat,attrs)
            #cleanup
            dsmod=dsmod.drop_vars([vname,vname_co])
    return dsmod


def deserialize(dsneq):
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
    return coo_deserialize(dsout)

def serialize(dsneq):
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
    return coo_serialize(dsneq)
    



