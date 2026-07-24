## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2026 Roelof Rietbroek, r.rietbroek@utwente.nl
import pandas as pd
from scipy.sparse import block_diag
from sparse import as_coo # needed to embed sparse arrays in xarray
import xarray as xr
from xinv.core.attrs import xinv_st,ltpl_attrs,sigma0_attrs,cov_attrs,islower
import numpy as np

class BlockBy:
    """
     A class to aid in operations on blocks of normal equations systems. Similar in philosophy to pandas  groupby
    """
    def __init__(self,ds_src,group_level):
        self._dssrc=ds_src
        self._grplev=group_level

        #get a list of unique values
        unkdim,unkdim_=self._dssrc.xi.unknown_dim()
        idx=self._dssrc.get_index(unkdim)
        if idx.name == self._grplev and type(idx) == pd.Index:
            #note when the values in the index are unique (likely) this only loops over the diagonals only
            self._grps=idx.unique()
        elif self._grplev in idx.names and type(idx) == pd.MultiIndex:
            self._grps=idx.get_level_values(self._grplev).unique()
        else:
            raise ValueError(f"{self._grplev} not found in system's unknown index")
    
    def __len__(self):
        return len(self._grps)

    def __iter__(self):
        for uniq_grp in self._grps:
            #extract a diagonal sub block from the source system
            yield uniq_grp,self._dssrc.xi.sel(**{self._grplev:uniq_grp})        
        
    def solve(self):
        """
            Solve the normal equation system, while ignoring the elements outside the diagonal blocks
        """
        

        Nvar=self._dssrc.xi.N
        Nstate=Nvar.attrs['xinv_state']
        unkdim,unkdim_=self._dssrc.xi.unknown_dim()
        
        outattr=cov_attrs(islower(Nvar))
        #determine block matrix state
        blockattr=Nvar.attrs.copy()
        #determine output matrix state
        
        if Nstate ==  xinv_st.BsymU:
            blockattr['xinv_state']=xinv_st.symU
            outattr['xinv_state']=xinv_st.BsymU
        elif Nstate ==  xinv_st.BsymL:
            blockattr['xinv_state']=xinv_st.symL
            outattr['xinv_state']=xinv_st.BsymL
        elif Nstate == xinv_st.symL:
            outattr['xinv_state']=xinv_st.BsymL
        elif Nstate == xinv_st.symU:
            outattr['xinv_state']=xinv_st.BsymU
        else:
            raise ValueError(f"Unrecognized normal matrix state {outattr['xinv_state']}")
        
        ltpl=self._dssrc.xi.ltpl.copy(deep=True) 
        deltaltpl=ltpl.copy(deep=True)
        deltaltpl.data[()]=0
        dsouts=[]
        blocks=[] 
        dropvars=None
        outidx=None
        
        for grp,dsub in self:
            #update attributes 
            dsub.xi.N.attrs=blockattr
            if hasattr(dsub.xi.N.data,'todense'):
                dsub.xi.N.data=dsub.xi.N.data.todense()
            
            dsblock=dsub.xi.solve()
            deltaltpl+=ltpl-dsblock.xi.ltpl
            if outidx is None:
                outidx=dsblock.get_index(unkdim)
            else:
                outidx=outidx.append(dsblock.get_index(unkdim))
            if dropvars is None:
                CVname=dsblock.xi.COV.name
                dropvars=[CVname,dsblock.xi.ltpl.name,dsblock.xi.sigma0.name,dsblock.xi.nobs.name,dsblock.xi.npara.name]
            blocks.append(dsblock[CVname].data)
            dsouts.append(dsblock.drop_vars(dropvars))
        #assemble result in large sparsei block diagonal system
        
        dsout=xr.concat(dsouts,dim=unkdim)
        
        dsout[ltpl.name]=ltpl-deltaltpl
        nobs=self._dssrc.xi.nobs
        npara=self._dssrc.xi.npara
        
        dsout[ltpl.name].attrs.update(ltpl_attrs('posteriori'))
        dsout['sigma0']=np.sqrt(dsout[ltpl.name]/(nobs-npara))
        dsout['sigma0'].attrs.update(sigma0_attrs('posteriori'))

        dsout[nobs.name]=nobs
        dsout[npara.name]=npara
        
            
        #add blockdiagonal matrix
        spmat=as_coo(block_diag(blocks,format='coo'))
        dsout[CVname]=([unkdim,unkdim_],spmat,outattr)
        return dsout

    def add(self,dsother):
        """
        Add another matrix to the block diagonal one while conversing the block diagional structure of the source normal  matrix (potentially discarding off diagonal elements)
        """
        

        outidx=None
        Nvar=self._dssrc.xi.N
        if Nvar is None:
            raise ValueError("Cannot find Normal matrix varaible")
        Nname=Nvar.name
        Nstate=Nvar.attrs['xinv_state']
        unkdim,unkdim_=self._dssrc.xi.unknown_dim()
        

        outattr=Nvar.attrs.copy()
        #determine block matrix state
        blockattr=Nvar.attrs.copy()
        #determine output matrix state
        
        if Nstate ==  xinv_st.BsymU:
            blockattr['xinv_state']=xinv_st.symU
        elif Nstate ==  xinv_st.BsymL:
            blockattr['xinv_state']=xinv_st.symL
        elif Nstate == xinv_st.symL:
            outattr['xinv_state']=xinv_st.BsymL
        elif Nstate == xinv_st.symU:
            outattr['xinv_state']=xinv_st.BsymU
        else:
            raise ValueError(f"Unrecognized normal matrix state {outattr['xinv_state']}")
        
        

        #create another blockby object and check for consistency
        blockother=dsother.xi.blockby(self._grplev)

        if len(self) != len(blockother):
            raise ValueError("inconsistent level of blocks, refusing to add")
        dsouts=[]
        blocks=[] 
        dropvars=None
        for (basegrp,dsbase),(othergrp,dsadd) in zip(self,blockother):
            if basegrp != othergrp:
                raise ValueError("block orders differ, cannot not add")
            #update attributes 
            dsbase.xi.N.attrs=blockattr
            if hasattr(dsbase.xi.N.data,'todense'):
                dsbase.xi.N.data=dsbase.xi.N.data.todense()
            
            dsadd.xi.N.attrs=blockattr
            if hasattr(dsadd.xi.N.data,'todense'):
                dsadd.xi.N.data=dsadd.xi.N.data.todense()
            
            dsblock=dsbase.xi.add(dsadd)
            if outidx is None:
                outidx=dsblock.get_index(unkdim)
            else:
                outidx=outidx.append(dsblock.get_index(unkdim))
            if dropvars is None:
                dropvars=[Nname,dsblock.xi.ltpl.name,dsblock.xi.sigma0.name,dsblock.xi.nobs.name,dsblock.xi.npara.name]
            blocks.append(dsblock[Nname].data)
            dsouts.append(dsblock.drop_vars(dropvars))
        #assemble result in large sparsei block diagonal system
        
        dsout=xr.concat(dsouts,dim=unkdim)
        
        sigma0=self._dssrc.xi.sigma0
        if sigma0:
            dsout[sigma0.name]=sigma0
        
        ltpl=self._dssrc.xi.ltpl
        if ltpl:
            dsout[ltpl.name]=ltpl+dsother.xi.ltpl

        nobs=self._dssrc.xi.nobs
        if nobs:
            dsout[nobs.name]=nobs+dsother.xi.nobs
        
        dsout[self._dssrc.xi.npara.name]=self._dssrc.xi.npara+dsother.xi.npara-dsout.sizes[unkdim]

            
        #add blockdiagonal matrix
        spmat=as_coo(block_diag(blocks,format='coo'))
        dsout[Nname]=([unkdim,unkdim_],spmat,outattr)
        return dsout

    def transform(self,fwdop,**kwargs):
        dsouts=[]
        blocks=[]
        outidx=None
        Nvar=self._dssrc.xi.N
        if Nvar is None:
            raise ValueError("Cannot find Normal matrix varaible")
        Nname=Nvar.name
        outattr=Nvar.attrs.copy()
        #determine output matrix state
        if Nvar.attrs['xinv_state'] ==  xinv_st.symU:
            outattr['xinv_state']=xinv_st.BsymU
        elif Nvar.attrs['xinv_state'] ==  xinv_st.symL:
            outattr['xinv_state']=xinv_st.BsymL
        elif Nvar.attrs['xinv_state'] in [xinv_st.BsymL,xinv_st.BsymU]:
            #find just pass
            pass
        else:
            raise ValueError(f"Unrecognized normal matrix state {outattr['xinv_state']}")
        dropvars=None

        npara=self._dssrc.xi.npara.copy(deep=True)
        for grp,dsub in self:
            dsblock=dsub.xi.transform(fwdop,**kwargs)
            npara[()]+=dsblock.xi.npara-dsub.xi.npara
            if outidx is None:
                outidx=dsblock.get_index(fwdop._unkdim)
            else:
                outidx=outidx.append(dsblock.get_index(fwdop._unkdim))
            if dropvars is None:
                dropvars=[Nname,dsblock.xi.ltpl.name,dsblock.xi.sigma0.name,dsblock.xi.nobs.name,dsblock.xi.npara.name]

            blocks.append(dsblock[Nname].data)
            dsouts.append(dsblock.drop_vars(dropvars))
        #assemble result in large sparsei block diagonal system
        
        
        dsout=xr.concat(dsouts,dim=fwdop._unkdim)
        #fix auxiliary parameters (e.g. npara,nobs,..)
        dsout[self._dssrc.xi.npara.name]=npara
        

        sigma0=self._dssrc.xi.sigma0
        if sigma0 is not None:
            dsout[sigma0.name]=sigma0
        
        ltpl=self._dssrc.xi.ltpl

        if ltpl is not None:
            dsout[ltpl.name]=ltpl

        nobs=self._dssrc.xi.nobs

        if nobs is not None:
            dsout[nobs.name]=nobs




        #add blockdiagonal matrix
        spmat=as_coo(block_diag(blocks,format='coo'))


        dsout[Nname]=([fwdop._unkdim,fwdop._unkdim+"_"],spmat,outattr)
        
        return dsout

        
