## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl


# This file contains NEQ attributes of the xinv package and lookup methods to determine normal equation components


def xinv_attrs(xitype,xistate,xidescr):
    return {"xinv_type":xitype,"xinv_state":xistate,"xinv_description":xidescr}

def rhs_attrs():
    return xinv_attrs("rhs",None,"right hand side vector")

def solest_attrs():
    return xinv_attrs("solest",None,"solution estimate")


def N_attrs(lower=0):
    if lower == 0:
        state="SymUpper"
    else:
        state="SymLower"
    return xinv_attrs("N",state,"Normal matrix")

def Chol_attrs(lower=0):
    if lower == 0:
        state="CholeskyUpper"
    else:
        state="CholeskyLower"
    return xinv_attrs("N",state,"Cholesky decomposition")

def cov_attrs(lower=0):
    if lower == 0:
        state="SymUpper"
    else:
        state="SymLower"
    return xinv_attrs("COV",state,"Error Covariance matrix")

def ltpl_attrs(state="apriori"):
    return xinv_attrs("ltPl",state,"Least squares cost function values")

def sigma0_attrs(state="apriori"):
    return xinv_attrs("sigma0",state,"apriori/posteriori standard deviation error scale")

def nobs_attrs():
    return xinv_attrs("nobs","initial","number of original observations")

def npara_attrs():
    return xinv_attrs("npara","initial","number of unknown parameters (explicit and implicit)")

def group_coord_attrs(state="unlinked"):
    return xinv_attrs("group_coord",state,"xinv coordinate group")

def find_neq_components(dsneq):
    components=["N","rhs","ltPl","sigma0","nobs","npara"]
    out=[]
    for comp in components:
        try:
            compname=[ky for ky in dsneq.keys() if dsneq[ky].attrs['xinv_type'] == comp ][0]
        
            out.append(dsneq[compname])
        except KeyError:
            raise KeyError(f"Cannot find {comp} component in NEQ dataset")

    return out

def find_group_coords(dsneq):
    groupcoords={}
    for coordname in dsneq.coords:
        if "xinv_type" in dsneq[coordname].attrs and dsneq[coordname].attrs['xinv_type'] == "group_coord":
            groupcoords[coordname]=dsneq[coordname]

    return groupcoords
