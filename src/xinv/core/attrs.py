## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl


# This file contains NEQ attributes of the xinv package and lookup methods to determine normal equation components
from collections import namedtuple
from xinv.core.logging import xinvlogger 

#centrally define the xinv attributes
_xinvtype=namedtuple("_xinvtype","unk_co aux_co grp_id_co grp_seq_co N COV rhs solest stdsolest x0 ltpl sigma0 nobs npara")

xinv_tp=_xinvtype("unk_coord","aux_coord","group_id_coord","group_seq_coord","N","COV", "rhs","solest","solest_std","aprioriVec","ltPl","sigma0","nobs","npara")


_xinvstate=namedtuple("_xinvstate","linked unlinked init apri post symU symL cholU cholL")
xinv_st=_xinvstate("linked","unlinked","initial","apriori","posteriori","SymUpper","SymLower","CholeskyUpper","CholeskyLower")


def xinv_attrs(xitype,xistate,xidescr):
    return {"xinv_type":xitype,"xinv_state":xistate,"xinv_description":xidescr}

def rhs_attrs():
    return xinv_attrs(xinv_tp.rhs,xinv_st.init,"right hand side vector")

def x0_attrs():
    return xinv_attrs(xinv_tp.x0,xinv_st.apri,"Apriori solution estimate")

def solest_attrs():
    return xinv_attrs(xinv_tp.solest,xinv_st.init,"solution estimate")

def stdsolest_attrs():
    return xinv_attrs(xinv_tp.stdsolest,xinv_st.init,"Standard deviations of the solution estimate")


def N_attrs(lower=0):
    if lower == 0:
        state=xinv_st.symU
    else:
        state=xinv_st.symL

    return xinv_attrs(xinv_tp.N,state,"Normal matrix")

def Chol_attrs(lower=0):
    if lower == 0:
        state=xinv_st.cholU
    else:
        state=xinv_st.cholL
    return xinv_attrs(xinv_tp.N,state,"Cholesky decomposition")

def cov_attrs(lower=0):
    if lower == 0:
        state=xinv_st.symU
    else:
        state=xinv_st.symL
    return xinv_attrs(xinv_tp.COV,state,"Error Covariance matrix")

def islower(mat):
    """Determine if the xinv_type_str is a lower triangular matrix"""
    state=mat.attrs['xinv_state']
    if state == xinv_st.symL or state == xinv_st.cholL :
        return 1
    elif state == xinv_st.symU or state == xinv_st.cholU :
        return 0
    else:
        raise ValueError(f"Unknown xinv state: {state}. Must be one of {xinv_st.symL}, {xinv_st.symU}, {xinv_st.cholL}, {xinv_st.cholU}")

def ltpl_attrs(state=xinv_st.apri):
    return xinv_attrs(xinv_tp.ltpl,state,"Least squares cost function values")

def sigma0_attrs(state=xinv_st.apri):
    return xinv_attrs(xinv_tp.sigma0,state,"apriori/posteriori standard deviation error scale")

def nobs_attrs():
    return xinv_attrs(xinv_tp.nobs,xinv_st.init,"number of original observations")

def npara_attrs():
    return xinv_attrs(xinv_tp.npara,xinv_st.init,"number of unknown parameters (explicit and implicit)")

def xunk_coords_attrs(state=xinv_st.unlinked):
    """

    Parameters
    ----------
    state : str, optional
        state of the coordinate (linked or unlinked). 'unlinked' denotes that the coordinate sequence may be out of sync with the actual coordinate
        
    Returns
    -------
    xinv_attrs : dict   
    a dictionary with xinv-compliant attributes
    """
    return xinv_attrs(xinv_tp.unk_co,state,"xinv unknown parameter coordinate")

def aux_coords_attrs(state=xinv_st.unlinked):
    return xinv_attrs(xinv_tp.aux_co,state,"xinv auxiliary right hand side/solest coordinate")

def group_id_attrs(state=xinv_st.unlinked):
    return xinv_attrs(xinv_tp.grp_id_co,state,"xinv unknown parameter coordinate indicating the group ids")

def group_seq_attrs(state=xinv_st.unlinked):
    return xinv_attrs(xinv_tp.grp_seq_co,state,"xinv unknown parameter coordinate indicating the sequence within a group")

def find_component(dsneq,component):
    components=[ky for ky in dsneq.keys()]+[ky for ky in dsneq.coords.keys()]
    try:
        compname=[ky for ky in components if "xinv_type" in dsneq[ky].attrs and dsneq[ky].attrs['xinv_type'] == component ]
        return dsneq[compname[0]]
    except KeyError:
        raise KeyError(f"Cannot find {component} component in NEQ dataset")
    except IndexError:
        raise KeyError(f"Cannot find {component} component in NEQ dataset")

def find_components(dsneq,components):
    out=[]
    for comp in components:
        try:
            comp_=find_component(dsneq,comp)
        except KeyError:
            xinvlogger.warning(f"Cannot find {comp} component in NEQ dataset, setting to None")
            comp_=None

        out.append(comp_)
    return out


def find_neq_components(dsneq):
    components=[xinv_tp.N,xinv_tp.rhs,xinv_tp.x0,xinv_tp.ltpl,xinv_tp.sigma0,xinv_tp.nobs,xinv_tp.npara]
    return find_components(dsneq,components)

def find_sol_components(dssol):
    components=[xinv_tp.COV,xinv_tp.solest,xinv_tp.ltpl,xinv_tp.sigma0,xinv_tp.nobs,xinv_tp.npara]
    return find_components(dssol,components)



def find_xinv_coords(dsneq,exclude=None,include=None,state=None):
    if include is not None and exclude is not None:
        raise ValueError("Cannot specify both include and exclude in the same call")

    xinvcoords={}
    coordtypes=[xinv_tp.unk_co,xinv_tp.aux_co,xinv_tp.grp_id_co,xinv_tp.grp_seq_co]
    if exclude is not None:
        #make a subselection
        coordtypes=[coordtype for coordtype in coordtypes if coordtype not in exclude]
    elif include is not None:
        #replace with includes only 
        coordtypes=include

    for coordname in dsneq.coords:
        if "xinv_type" in dsneq[coordname].attrs and dsneq[coordname].attrs['xinv_type'] in coordtypes:
            if state is not None and dsneq[coordname].attrs['xinv_state'] != state:
                continue
            xinvcoords[coordname]=dsneq[coordname]

    return xinvcoords

def find_xinv_unk_coord(dsneq):

    xunk_co=find_xinv_coords(dsneq,include=[xinv_tp.unk_co],state=xinv_st.linked)
    if len(xunk_co)> 1:
        raise ValueError("Ambiguous linked unknown coordinate found")
    return next(iter(xunk_co.values()))

def find_xinv_group_coords(dsneq):
    """
    Find the linked group id and sequence coordinates in a xinv dataset

    Parameters
    ----------
    dsneq : xarray.Dataset
        The dataset to search in
    Returns
    -------
    group_id_co,group_seq_co : xarray.DataArray
        The group id and sequence coordinates, or None if not found
    """

    grp_co=find_xinv_coords(dsneq,include=[xinv_tp.grp_id_co,xinv_tp.grp_seq_co],state=xinv_st.linked)
    grpid_co=None
    grpseq_co=None
    for ky,co in grp_co.items():
        if co.attrs['xinv_type'] == xinv_tp.grp_id_co:
            grpid_co=co
        elif co.attrs['xinv_type'] == xinv_tp.grp_seq_co:
            grpseq_co=co

    return grpid_co,grpseq_co


def get_xunk_size_coname(dsneq):
    """Retrieve the size and name of the currently linked unknown coordinate
    """
    xunk_co=find_xinv_coords(dsneq,include=[xinv_tp.unk_co],state=xinv_st.linked)
    xunkconame=next(iter(xunk_co))
    return dsneq.sizes[xunk_co[xunkconame].dims[0]],xunkconame

def unlink(davar):
    """
    Unlink a variable by changing its state to unlinked
    Parameters
    ----------
    davar : xarray.DataArray
        The data array to unlink.

    Returns
    -------
    None.
    """
    change_state(davar,xinv_st.unlinked)

def link(davar):
    """
    Link a variable by changing its state to unlinked
    Parameters
    ----------
    davar : xarray.DataArray
        The data array to unlink.

    Returns
    -------
    None.
    """
    change_state(davar,xinv_st.linked)

def change_state(davar,state):
    """
    Change the state of a variable to a new state
    Parameters
    ----------
    davar : xarray.DataArray
        The data array to change the state of.
    state : str
        The new state to set.

    Returns
    -------
    None.
    """
    if "xinv_state" in davar.attrs:
        davar.attrs["xinv_state"]=state


def get_state(davar):
    """
    Get the state of a coordinate variable
    Parameters
    ----------
    davar : xarray.DataArray
        The data array to get the state of.

    Returns
    -------
    xinv_st
        The state of the data array.
    """
    if "xinv_state" in davar.attrs:
        return davar.attrs["xinv_state"]
    else:
        raise ValueError("No xinv_state attribute found")

def get_type(davar):
    """
    Get the type of a coordinate variable
    Parameters
    ----------
    davar : xarray.DataArray
        The data array to get the state of.

    Returns
    -------
    xinv_st
        The state of the data array.
    """
    if "xinv_type" in davar.attrs:
        return davar.attrs["xinv_type"]
    else:
        raise ValueError("No xinv_type attribute found")
