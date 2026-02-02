## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl

import xarray as xr
from xinv.core.attrs import xinv_tp,xinv_st,REG_attrs,xunk_coords_attrs,alpha_attrs,find_component,islower
from xinv.core.grouping import find_group_coords
from xinv.core.logging import xinvlogger
import numpy as np
from sparse import eye
from xinv.core.tools import find_ilocs

def regSys(dain:xr.DataArray,lower=0):
    """
    Wrap and decorate a square symmetric input matrix so it can be used as a regularization system.
    A regularization system is essentially a simplified Normal Equation System
    
    Parameters
    ----------
    dain: xarray.DataArray
        Square input matrix to be wrapped as a regularization system
    lower:int, default=0    
        Whether to register the upper (lower=0) or lower triangle(lower=1) of the matrix

    Returns
    -------
    An decorated Regularization matrix as a xarray.Dataset

    """
    
    #sanity checks
    sz=dain.shape[0]

    if len(dain.shape) != 2 or dain.shape[1] != sz:
        raise RunTimeError(f"regSys: input is not square {dain.shapes}")


    if dain.name is None:
        name=xinv_tp.REG
    else:
        name=dain.name

    dsout=dain.to_dataset(name=name)
    
    #check if a coordinate on the second dimension exist and drop ity whne consistent
    if dain.dims[1] in dsout.coords:
        if np.all(dsout.coords[dain.dims[0]].data == dsout.coords[dain.dims[1]].data):        
            dsout=dsout.drop_vars([dain.dims[1]])
        else:
            raise RunTimeError("Coordinates on the sides are not equal, refusing to proceed")

    dsout[name].attrs.update(REG_attrs(lower))
    
    #update attributes of the coordinates
    try:
        dsout.coords[dsout[name].dims[0]].attrs.update(xunk_coords_attrs(xinv_st.linked))
    except KeyError:
        xinvlogger.warning("Missing Coordinate variable in input array, continueing nevertheless")
    
    

    #add alpha=1.0
    dsout['alpha']=1.0
    dsout.alpha.attrs.update(alpha_attrs())

    return dsout




def getTikhonov(*, alpha=1.0,**kwargs):
    """
    Return a unit diagonal (Tikhonov) regularization matrix, with an initial scaling
    
    Parameters
    ----------
    alpha: float, default=1.0
        Initial scale of the Tikhonov regularization matrix
    **kwargs: 
        List of coordinate parameters to constrain. Specify as coordinatename=[1,2,4...]
    Returns
    -------
        A xarray.Dataset holding a sparse unit diagonal matrix and decorated as a regularization matrix
        
    """
    
    if len(kwargs) != 1:
        raise ValueError("No or too many arguments provided")

    name,param=next(iter(kwargs.items()))

    sz=len(param)
    
    #create a dataarray
    datik=xr.DataArray(eye(sz),dims=(name,name+'_'),coords={name:(name,param)},name="tik")
    #convert into a regularizatiobn  system
    return regSys(datik)


def regadd(dsneq:xr.Dataset,dsreg:xr.Dataset,alpha=None,inplace=False):
    """
        Adds a regularization to a  normal equation system and returns the resulting normal equation system
    """
    if alpha is None:
        alpha=dsreg.alpha.item()
    
    if not inplace:
        dsneq=dsneq.xi.deepcopy()
    
    # find normal matrix of the input
    try:
        N=find_component(dsneq,xinv_tp.N)
        sigma0=find_component(dsneq,xinv_tp.sigma0)
        #check for common sigma (required)
        if sigma0.shape:
            #they all need to be equal

            np.all(sigma0.data[0] == sigma0.data)
            sigma=sigma0.data[0]
        else:
            sigma=sigma0.item()
    except:
        raise ValueError("Cannot find normal matrix in the input")
    
    # find regularization matrix of the input
    try:
        R=find_component(dsreg,xinv_tp.REG)
    except:
        raise ValueError("Cannot find regularization matrix in the input")
    
    unkdim=N.dims[0]

    unkdimreg=R.dims[0]
    

    # find location

    #check for possible conflicts in the subgroups  
    group_id_co,group_seq_co,group_assoc=find_group_coords(dsneq)
    if group_id_co is not None and group_seq_co is not None:
        #check if 
        r_group_id,r_group_seq,r_asso=find_group_coords(dsreg)
        if r_group_id is None and r_group_seq is None:
            dsreg=dsreg.xi.as_group({unkdimreg:unkdim},lookup_co=dsneq)
            #refind the regularization matrix
            R=find_component(dsreg,xinv_tp.REG)
            unkdimreg=R.dims[0]
        else:
            #check for consistency in the associated coordinates
            for coname,co in r_asso.items():
                if not co.equals(group_assoc[coname]):
                    raise RuntimeError("Inconsistent associated coordinates between regularization and normal equation")

    if unkdim != unkdimreg:
        xinvlogger.warning("Dimensions names do not match, trying anyway")
    try:
        idxr=find_ilocs(dsneq,unkdim,R[unkdimreg].data)
    
    except KeyError:
        # extract a subset of the matrix and issue a warning about unused parameters


        raise KeyError("Regularization coordinate contain values not found in the normal equation system")



    # possibly expand sparse regularization matrix (can be improved/specialized when performance requires it)
    if hasattr(R.data,'todense'):
        R.data=R.data.todense()
    scale=np.power(sigma ,2) * alpha
    #add scaled regularization matrix inplace
    if islower(N) == islower(R):
        N[idxr,idxr]+=scale*R.data
    else:
        #add transpose
        N[idxr,idxr]+=scale*R.data.T
    return dsneq
