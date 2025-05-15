## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, rietbroek@utwente.nl

import numpy as np
from scipy.linalg.blas import dsyrk
import xarray as xr
from xinv.core.attrs import rhs_attrs,N_attrs,ltpl_attrs,sigma0_attrs,nobs_attrs,npara_attrs,find_xinv_coords, xunk_coords_attrs,aux_coords_attrs



class FwdOpbase:
    def __init__(self,obs_dim=None,unknown_dim=None,cache=False,jacobname="jacobian"):
        self._obsdim=obs_dim
        self._unkdim=unknown_dim
        self._unkdim_t=unknown_dim+"_"
        self._daobs=None
        self._cache_jacobian=cache
        self._jacob=None
        self._jacobname=jacobname

    def jacobian(self,daobs=None): 
        """Create the Jacobian of the forward operator"""
        if daobs is None and self._jacob is None:
            raise ValueError("Requesting the Jacobian without arguments requires caching abilities of the forward operator")
        elif daobs is None:
            return self._jacob

        jacob=self._jacobian_impl(daobs)
        if type(jacob) == xr.DataArray:
            jacob.name=self._jacobname
            jacob=jacob.to_dataset()
        #add xunk_coord attributes to possible coordinate
        if self._unkdim in jacob.coords:
            jacob[self._unkdim].attrs.update(xunk_coords_attrs(state='linked'))
        

        if self._cache_jacobian:
            self._jacob=jacob
        return jacob

    def build_normal(self,daobs,ecov=1):
        
        if len(daobs.sizes) > 2:
            raise NotImplementedError("Cannot handle multiple auxiliary dimensions at the moment")
        

        
        #create a design matrix for the given observations
        dsdesign=self.jacobian(daobs)
         
        dadesign=dsdesign[self._jacobname]

        if daobs.dims[0] != self._obsdim:
            #garantee that the observation dimension is the first dimension
            daobs=daobs.T

        nrhs=daobs.shape[1]
        nrhsdim=daobs.dims[1]

        if not np.isscalar(ecov):
            #decorrelate  error covariance
            dadesign=ecov.decorrelate(dadesign) 
            daobs=ecov.decorrelate(daobs)
            sigma0=np.ones(nrhs)
        else:        
            sigma0=np.ones(nrhs)*ecov
        
        #create the normal equation system (not only upper triangle filled
        lower=0
        trans=1
        if dadesign.data.flags['C_CONTIGUOUS'] and dadesign.data.strides[1] == 8:
            pass
        elif dadesign.data.flags['F_CONTIGUOUS'] and dadesign.data.strides[0] == 8:
            pass
        else:
            raise RuntimeError("Design matrix is not C or F contiguous")
        normal_matrix=dsyrk(1.0,dadesign.data,trans=trans,lower=lower)

        #create the right hand side vector(s)
        rhs=dadesign.T@daobs

        #allow for multiple right hand sides
        ltpl=daobs.dot(daobs,dim=self._obsdim)
        
        #set number of observations and unknown parameters (ensure integer type)
        nobs=daobs.sizes[self._obsdim]*np.ones(nrhs,dtype=np.int64)
        npara=normal_matrix.shape[0]*np.ones(nrhs,dtype=np.int64)
        
        #tag auxiliary coords in rhs as xinv_types
        if len(rhs.shape) > 1: 
            auxdim=[dim for dim in rhs.dims if dim != self._unkdim][0]
            #add auxialiary_coord attributes to possible coordinate
            rhs.coords[auxdim].attrs.update(aux_coords_attrs(state='linked'))
        
        xinvcoords=find_xinv_coords(dsdesign)
        #update with the relevant coordinates from the righthand side matrix
        xinvcoords.update({name:coord for name,coord in rhs.coords.items() if name not in xinvcoords.keys()})
        dsout=xr.Dataset(dict(N=((self._unkdim,self._unkdim_t),normal_matrix.data),rhs=(rhs.dims,rhs.data),ltpl=((nrhsdim),ltpl.data),sigma0=((nrhsdim),sigma0.data),nobs=((nrhsdim),nobs.data),npara=((nrhsdim),npara.data)),coords=xinvcoords)
        #for some reason the coordinate attributes do not get properly propagated
        #so make sure they are added
        for key,coord in xinvcoords.items():
            dsout[key].attrs.update(coord.attrs)

        #add attributes
        dsout.N.attrs.update(N_attrs(lower))
        dsout.rhs.attrs.update(rhs_attrs())
        dsout.ltpl.attrs.update(ltpl_attrs('apriori'))
        dsout.sigma0.attrs.update(sigma0_attrs('apriori'))
        dsout.nobs.attrs.update(nobs_attrs())
        dsout.npara.attrs.update(npara_attrs())
        return dsout
    
    def __call__(self,inpara):
        """Apply the forward operator"""
        return self.jacobian()[self._jacobname]@inpara
