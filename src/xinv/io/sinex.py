## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl
## Read Normal equation systems or solutions from SINEX (2.02) files
## This is currently not feature complete



import numpy as np
import xarray as xr
from datetime import datetime,timedelta
from xinv.core.grouping import build_group_coord
from xinv.core.logging import xinvlogger
from xinv.io.gzipwrap import gzip_open_r
from xinv.xiext import  read_symmat_fast
from xinv.core.attrs import * 
import pandas as pd

snxunkdim="snx_unk"

snxunkco=f"_{snxunkdim}_idx"

class SNXBlock:
    """
    Base class to parse various SINEX blocks
    """
    def __init__(self,blockname):
        self.block=blockname
    def parseblock(self,fileobj,dsout):
        pass



def sinex2date(snxdate:str)->datetime:
    """
        Convert sinex datestring in yy:doy:seconds to python datetime

    Parameters
    ----------
    snxdate : str
        datestring in sinex format
        

    Returns
    -------
    datetime
       Specified date and time as datetime object 

    """
    yr,doy,sec=[int(x) for x in snxdate.split(":")]
    return datetime(yr+2000 if yr<50 else yr+1900,1,1)+timedelta(days=doy-1,seconds=sec)


def read_symmat_slow(fileobj,dsout,blockname):
    """
        Reads a triangular matrix from a SINEX block and returns a symmetric version.
        This is a slow Pure Python implmentation, use read_symmat_fast for a faster cythonized version
    Parameters
    ----------
    fileobj : 
        io buffer to read lines from
        
    dsout : xarray.Dataset
        xarray.Dataset to augment the matrix data to
        
        
    blockname : str
        name of the SINEX block. Should be one of:
        SOLUTION/NORMAL_EQUATION_MATRIX U
        SOLUTION/NORMAL_EQUATION_MATRIX L
        

    Returns
    -------
    an updated xarray.Dataset holding the new matrix in a new variable 'N'

    """
    if not blockname.startswith('SOLUTION/NORMAL_EQUATION_MATRIX'):
        raise RuntimeError(f"Wrong block {blockname}?")
    nest,xunk_co=get_xunk_size_coname(dsout)
    xunkdim=dsout[xunk_co].dims[0]
    mat=np.zeros([nest,nest],order='C')
    data=np.zeros([3])
    for line in fileobj:
        if line[0] == '-':
            #end of block encountered
            break
        elif line[0] =='*':
            #comment
            continue
        data=[float(x) for x in line.split()]
        irow=int(data[0])-1 #note zero indexing
        icol=int(data[1])-1
        ndat=len(data)-2
        mat[irow,icol:icol+ndat]=data[2:]
    #mirror the upper triangle in the lower part
    mat=np.triu(mat,k=1).T+mat

    # if "nm_" not in dsout.indexes:
        # #add the transposed index
        # mi_=SHindexBase.mi_toggle(dsout.indexes['nm'])
        # dsout=dsout.sh.set_nmindex(mi_,'_')

    dsout['N']=([xunkdim,xunkdim+'_'],mat)
    dsout.N.attrs.update(N_attrs())
    return dsout

class SNXVEC(SNXBlock):
    def __init__(self,blockname):
        super().__init__(blockname)
        self._dispatch=dict(CN=self.CNparse,SN=self.CNparse,STAX=self.STA_parse,STAY=self.STA_parse,STAZ=self.STA_parse)
        self._coords={}
        self._coord_multindex_names={}
        if blockname == "SOLUTION/APRIORI":
            self.vname="apri_est"
            self.svname=None
            self.vattrs=x0_attrs()
            self.svattrs={}
        elif blockname == "SOLUTION/NORMAL_EQUATION_VECTOR":
            self.vname="rhs"
            self.svname=None
            self.vattrs=x0_attrs()
            self.svattrs={}
        elif blockname == "SOLUTION/ESTIMATE":
            self.vname="sol_est"
            self.svname="sol_std"
            self.vattrs=solest_attrs()
            self.svattrs=stdsolest_attrs()
        else:
            raise NotImplementedError(f"{blockname} not recognized")


    def parseblock(self, fileobj, dsout):
         
        
        #allocate space
        nest,unkco=get_xunk_size_coname(dsout)
        unkdim=dsout[unkco].dims[0]
        #allocate space
        values=np.zeros([nest])
        if self.svname is not None:
            stddevs=np.zeros([nest])

        refepochs=np.empty([nest],dtype=datetime)
        grp_co=np.empty([nest],dtype=object)

        for line in fileobj:
            if line.startswith('-'):
                #end of block encountered
                break
            elif line.startswith('*'):
                #comment
                continue

            #find the parameter group, and line parser from the type
            snxtype=line[7:7+6].strip()
            idx,grpidseq,refepoch,val,std=self._dispatch.get(snxtype,self.generic_parse)(line)
            refepochs[idx]=refepoch
            values[idx]=val
            if std:
                stddevs[idx]=std
            grp_co[idx]=grpidseq
        dsout[self.vname]=(unkdim,values)
        dsout[self.vname].attrs.update(self.vattrs)
        
        if self.svname:
            dsout[self.svname]=(unkdim,stddevs)
            dsout[self.svname].attrs.update(self.svattrs)
        
        dsout['refepochs']=([unkdim],refepochs)
        
        #possibly add coordinates
        dsout=self.add_coords(dsout,grp_co)

        return dsout
    
    def add_coords(self,dsout,grp_co):
        
        nest,unkco=get_xunk_size_coname(dsout)
        unkdim=dsout[unkco].dims[0]
        if unkco != snxunkco:
            #no need to re-add coordinates, because tghey were already assigned in a prvious call of this function
            return dsout

        #change the state of the current unknown coordinate to unlinked
        change_state(dsout[snxunkco],xinv_st.unlinked)


        if len(self._coords) == 1:
            #only one group
            grpid=next(iter(self._coords))
            dsout=dsout.rename({unkdim:grpid})
            unkdim=grpid
            state=xinv_st.linked

        else:
            #multiple groups: add coordinates but in an unlinked state
            state=xinv_st.unlinked

        for grpid,coord in self._coords.items():

            if grpid  in ["nm","stat"]:
                #Modify the coordinateinto a multindex
                coord=pd.MultiIndex.from_tuples(coord, names=self._coord_multindex_names[grpid])
                coord=xr.Coordinates.from_pandas_multiindex(coord, grpid)
                dsout=dsout.assign_coords(coord)
                dsout[grpid].attrs.update(xunk_coords_attrs(state=state))
            else:
                #generic case
                dsout=dsout.assign_coords({grpid:(grpid,coord)})
                dsout[grpid].attrs.update(xunk_coords_attrs(state=state))
        
        #possibly add group_id and seq coordinates 
        if state == xinv_st.unlinked:
            dsout=dsout.assign_coords(build_group_coord(grp_co,dim=unkdim))

        return dsout

    def CNparse(self,line):
        grpid="nm"
        if grpid not in self._coords:
            #initialize subgroup
            self._coords[grpid]=[]
            self._coord_multindex_names[grpid]=["n","m"]

        fields = line.split()
        idx=int(fields[0])-1
        n = int(fields[2])
        m=int(fields[4])
        m=m if fields[1] == "CN" else -m
        
        self._coords[grpid].append((n,m))

        epoch=sinex2date(fields[5])
        val=float(fields[8])
        if self.svname:
            std=float(fields[9])
        else:
            std=None

         
        return idx,(grpid,len(self._coords[grpid])-1),epoch,val,std 
        

    def generic_parse(self,line):
        grpid="snxaux"

        if grpid not in self._coords:
            #initialize subgroup
            self._coords[grpid]=[]
            self._coord_multindex_names[grpid]=None
        idx=int(line[0:6])-1
        para=line[7:26]
        self._coords[grpid].append(para)
        epoch=sinex2date(line[27:27+12])
        val=float(line[46:46+22])

        if self.svname:
            std=float(line[69:69+22])
        else:
            std=None
        return idx,(grpid,len(self._coords[grpid])-1),epoch,val,std 


    def STA_parse(self,line):
        grpid="stat"

        if grpid not in self._coords:
            #initialize subgroup
            self._coords[grpid]=[]
            self._coord_multindex_names[grpid]=['statype','site','monument','soluid']
        idx=int(line[0:6])-1
        statype=line[7:7+6].strip()
        site=line[14:14+4]
        monument=line[19:19+2].strip()
        soluid=int(line[22:22+4])
                
        self._coords[grpid].append((statype,site,monument,soluid))
        epoch=sinex2date(line[27:27+12])
        val=float(line[46:46+22])

        if self.svname:
            std=float(line[69:69+22])
        else:
            std=None
        return idx,(grpid,len(self._coords[grpid])-1),epoch,val,std 




class STATISTICS(SNXBlock):
    def __init__(self,blockname):
        assert blockname == "SOLUTION/STATISTICS"
        super().__init__("SOLUTION/STATISTICS")

    def parseblock(self,fileobj,dsout):
        """
            Reads a SINEX block holding statistical metrics
        Parameters
        ----------
        fileobj : 
            io buffer to read lines from
            
        dsout : xarray.Dataset
            xarray.Dataset to augment the matrix data to
            
        Returns
        -------
        an updated xarray.Dataset holding the available statistics  as scalar variables


        """
        for line in fileobj:
            if line.startswith('-'):
                #end of block encountered
                break
            elif line.startswith('*'):
                #comment
                continue
                continue
            
            attrs=None

            if line.startswith(" NUMBER OF DEGREES OF FREEDOM"):
                varname="dof"
                tp=int
            elif line.startswith(" NUMBER OF OBSERVATIONS"):
                varname="nobs"
                tp=int
                attrs=nobs_attrs()
            elif line.startswith(" NUMBER OF UNKNOWNS"):
                varname="nunknown"
                tp=int
                attrs=npara_attrs()
            elif line.startswith(" WEIGHTED SQUARE SUM OF O-C"):
                varname="ltpl"
                tp=float
                attrs=ltpl_attrs()

            elif line.startswith(" VARIANCE FACTOR"):
                varname="sigma0_2"
                tp=float
                attrs=sigma0_attrs()
            elif line.startswith(" SQUARE SUM OF RESIDUALS (VTPV)"):
                varname="ltpl"
                tp=float
                attrs=ltpl_attrs()
            else:
                tp=None
                varname =None
                xinvlogger.warning(f"ignoring {self.block} entry {line}")

            if varname:
                spl = line.split()
                val=tp(spl[-1])
                if varname == "sigma0_2":
                    #convert to sigma0
                    val=np.sqrt(val)
                    varname="sigma0"
                dsout[varname]=val
                if attrs:
                    dsout[varname].attrs.update(attrs)
        return dsout


class FILEREF(SNXBlock):
    def __init__(self,blockname):
        assert blockname == "FILE/REFERENCE"
        super().__init__("FILE/REFERENCE")

    def parseblock(self,fileobj,dsout):
        globattrs={}
        for line in fileobj:
            if line.startswith('-'):
                #end of block encountered
                break
            elif line.startswith('*'):
                #comment
                continue
            ky=line[1:19]
            val=line[20:-1]
            globattrs["SNX-"+ky]=val
        dsout.attrs.update(globattrs)
        return dsout

class SYMMAT(SNXBlock):
    def __init__(self,blockname):
        super().__init__(blockname)

    def parseblock(self,fileobj,dsout):
        return read_symmat_fast(fileobj,dsout,self.block)
        # return read_symmat(fileobj,dsout,self.block)



# dictionary to lookup blockparser classes)

blockdispatch={"SOLUTION/STATISTICS":STATISTICS,
               "FILE/REFERENCE":FILEREF,
               "SOLUTION/APRIORI":SNXVEC,
               "SOLUTION/ESTIMATE":SNXVEC,
               "SOLUTION/NORMAL_EQUATION_VECTOR":SNXVEC,
               "SOLUTION/NORMAL_EQUATION_MATRIX U":SYMMAT,
               "SOLUTION/NORMAL_EQUATION_MATRIX L":SYMMAT}

compatversions=["2.02","2.01"]

def parse_headerline(line):
    if not line.startswith('%=SNX'):
        raise RuntimeError("Not a valid SINEX file")
    
    header={}
    header['version']=line[6:6+4]

    headerfields=line.split()
    if header['version'] not in compatversions:
        raise RuntimeError(f"read_sinex is not compatible with {header['version']}")
    
    header['nest']=int(line[60:60+5])

    header['tstart']=sinex2date(line[32:32+12])
    header['tend']=sinex2date(line[45:45+12])
    return header

def read_sinex(file_or_obj,stopatmat=False):
    """
        Reads normal equation information from a SINEX file (currently only partially supported)
    Parameters
    ----------
    file_or_obj : 
        IO buffer or filename with the SINEX data source
        
    stopatmat : bool
        Stop reading from the source when encountering a MATRIX block to speed up. (default=False)
        

    Returns
    -------
       a xarray.Dataset holding the normal equation information 
        

    """
    needsClosing=False
    if type(file_or_obj) == str:
        needsClosing=True
        if file_or_obj.endswith('.gz') or file_or_obj.endswith('.Z'): 
            file_or_obj=gzip_open_r(file_or_obj,textmode=True)
        else:
            file_or_obj=open(file_or_obj,'rt')
    
    line=file_or_obj.readline()
    header=parse_headerline(line)
    #initialize xarray dataset with some scalar vars to augment
    nest=header['nest']

    dsout=xr.Dataset(dict(tstart=header['tstart'],tend=header['tend']),coords={snxunkco:(snxunkdim,np.arange(nest))})
    # use  snx_ids as the unknown coordinate dimension (for now)
    dsout[snxunkco].attrs.update(xunk_coords_attrs(state=xinv_st.linked))
    
    # loop until a block is encountered and then dispatch to appropriate function
    for line in file_or_obj:
        if line[0] == "+":
            block=line[1:].strip()

            if stopatmat and "MATRIX" in block:
                xinvlogger.info(f"Encountered {block}, stopping")
                break
            if block not in blockdispatch.keys():
                xinvlogger.warning(f"Ignoring block {block}")
                continue
            xinvlogger.info(f"Reading block {block}")
            
            blockobj=blockdispatch[block](block)
            dsout=blockobj.parseblock(file_or_obj,dsout)

        
    if needsClosing:
        file_or_obj.close()

    return dsout 
    
