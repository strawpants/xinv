# Test some basic operations of xarray objects filled with spherical harmonic datasets
# This file is part of the shxarray software which is licensed
# under the Apache License version 2.0 (see the LICENSE file in the main repository)
# Copyright Roelof Rietbroek (r.rietbroek@utwente.nl), 2023


import pytest
import xarray as xr
import os
import numpy as np
import time
from xinv.core.logging import xinvlogger



@pytest.fixture
def sinexvaldata(request):
    if request.param == "ITSG":
        # some validation values based on prior visual inspection of sinex file
        #+SOLUTION/ESTIMATE
        # ...
        # 91 CN        9 --    7 03:259:00000 ---- 2 -1.17979570334447e-07 1.01491e-12
        # 92 SN        9 --    7 03:259:00000 ---- 2 -9.69270369824925e-08 1.02558e-12
        # ..
        #+SOLUTION/APRIORI
        #...
        # 25 CN        5 --    2 03:259:00000 ---- 2  6.52120740523400e-07 0.00000e+00
        # 26 SN        5 --    2 03:259:00000 ---- 2 -3.23349434999185e-07 0.00000e+00
        #..
        #+SOLUTION/NORMAL_EQUATION_VECTOR
        #*INDEX _TYPE_ CODE PT SOLN _REF_EPOCH__ UNIT S ___RIGHT_HAND_SIDE___
        #1 CN        2 --    0 03:259:00000 ---- 2 -3.27811136401904e+12
        #2 CN        2 --    1 03:259:00000 ---- 2 -4.32367855180938e+11
        #+SOLUTION/NORMAL_EQUATION_MATRIX U
        # ....
        # 7     7  2.92598076849103e+23 -3.55351048890917e+21 -3.78432268065340e+21
        # 7    10 -6.34992891236174e+21  2.34638506385098e+20  4.78761459861900e+21
        # ...
        valdata={} 
        valdata["sol_est"]=[(dict(n=9,m=7), -1.17979570334447e-07),(dict(n=9,m=-7),-9.69270369824925e-08)] 
        valdata["sol_std"]=[(dict(n=9,m=7),1.01491e-12),(dict(n=9,m=-7), 1.02558e-12)]
        valdata["apri_est"]=[(dict(n=5,m=2),6.52120740523400e-07),(dict(n=5,m=-2),-3.23349434999185e-07)] 
        valdata["rhs"]=[(dict(n=2,m=0),-3.27811136401904e+12),(dict(n=2,m=1),-4.32367855180938e+11)]
        valdata["N"]=[(7,7, 2.92598076849103e+23),(7,8,-3.55351048890917e+21),(7,9,-3.78432268065340e+21),(7,10,-6.34992891236174e+21),(7,11,2.34638506385098e+20),(7,12,4.78761459861900e+21)]
        
    #possibly also download the file and pass the file name

        url="https://ftp.tugraz.at/outgoing/ITSG/GRACE/ITSG-Grace2018/monthly/normals_SINEX/monthly_n96/ITSG-Grace2018_n96_2003-09.snx.gz"
    elif request.param == "IGS":
        # 36 STAZ   ALBH  A    3 22:351:43185 m    1 0.474579124249256E+07 .318348E-03
    # 37 STAX   ALGO  A    1 22:349:43185 m    1 0.918129084134536E+06 .184114E-03
    # 38 STAY   ALGO  A    1 22:349:43185 m    1 -.434607133252970E+07 .372889E-03
        valdata={} 
        valdata["sol_est"]=[(dict(statype="STAZ",site="ALBH",monument="A",soluid=3), 0.474579124249256E+07),
                            (dict(statype="STAX",site="ALGO",monument="A",soluid=1), 0.918129084134536E+06),
                            (dict(statype="STAY",site="ALGO",monument="A",soluid=1), -0.434607133252970E+07)]

        valdata["sol_std"]=[(dict(statype="STAZ",site="ALBH",monument="A",soluid=3), 0.318348E-03),
                            (dict(statype="STAX",site="ALGO",monument="A",soluid=1), 0.184114E-03),
                            (dict(statype="STAY",site="ALGO",monument="A",soluid=1), 0.372889E-03)]
    # 32 STAY   ALBH  A    2 22:350:43185 m    1 -.353904953666000E+07 .000000E+00
    # 33 STAZ   ALBH  A    2 22:350:43185 m    1 0.474579123972000E+07 .000000E+00
    # 34 STAX   ALBH  A    3 22:351:43185 m    1 -.234133313693707E+07 .000000E+00
        valdata["apri_est"]=[(dict(statype="STAY",site="ALBH",monument="A",soluid=2), -0.353904953666000E+07),
                            (dict(statype="STAZ",site="ALBH",monument="A",soluid=2), 0.474579123972000E+07),
                            (dict(statype="STAX",site="ALBH",monument="A",soluid=3), -0.234133313693707E+07)]
    # 2433 STAZ   ZIM3  A    3 22:351:43185 m    2 -.190963469354183E-01
    # 2434 STAX   ZIMM  A    1 22:351:43185 m    2 0.622005988863635E-02
    # 2435 STAY   ZIMM  A    1 22:351:43185 m    2 -.455479611760775E-01
        valdata["rhs"]=[(dict(statype="STAZ",site="ZIM3",monument="A",soluid=3), -0.190963469354183E-01),
                            (dict(statype="STAX",site="ZIMM",monument="A",soluid=1), 0.622005988863635E-02),
                            (dict(statype="STAY",site="ZIMM",monument="A",soluid=1), -0.455479611760775E-01)]
    # 46    40 -0.40477609814186E-02 -0.14917111114273E-02 -0.16167305540122E-02
    # 46    43 -0.40488191574758E-02 -0.54300924437564E-03  0.16845406308024E-02
    # 46    46  0.32043242340174E+02

        valdata["N"]=[(46,40, -0.40477609814186E-02),(46,41,-0.14917111114273E-02),(46,42,-0.16167305540122E-02),
                      (46,43, -0.40488191574758E-02),(46,44,-0.54300924437564E-03),(46,45, 0.16845406308024E-02),
                      (46,46,0.32043242340174E+02)]

        url="https://igs.bkg.bund.de/root_ftp/IGSac/products/2240/COD0OPSFIN_20223500000_01D_01D_SOL.SNX.gz"



    sinexfile=os.path.join(os.path.dirname(__file__),'testdata', os.path.basename(url))
    
    if not os.path.exists(sinexfile):
        import requests
        xinvlogger.info(f"Downloading {sinexfile}...")
        r=requests.get(url)
        with open(sinexfile,'wb') as fid:
            fid.write(r.content)
    return valdata,sinexfile


     

@pytest.mark.parametrize("sinexvaldata",["IGS","ITSG"],indirect=True)
def test_sinex(sinexvaldata):
    """ Test reading of Normal equation systems in SINEX format"""
    sinexval,sinexfile=sinexvaldata
    #Engine does not need to be specified because file corresponds to commonly used filename pattern for sinex 
    start=time.time()
    dsneqsinex=xr.open_dataset(sinexfile,drop_variables=["N"])
    end=time.time()
    xinvlogger.info(f"Time to read sinex file without normal matrix {end-start:.2f} seconds")

    if 'COD0OPSFI' in sinexfile:
        #get a subgrpup
        dsneqsinex=dsneqsinex.xi.get_group("stat")

    for var in ["sol_est","apri_est","sol_std","rhs"]:
        for midict,val in sinexval[var]:
            assert val == dsneqsinex[var].sel(midict).item()
        # assert dsneqsinex[var].sel(n=
    
    #read version with entire normal matrix
    start=time.time()
    dsneqsinex=xr.open_dataset(sinexfile)
    end=time.time()
    xinvlogger.info(f"Time to read sinex file with normal matrix {end-start:.2f} seconds")
    for ix,iy,val in sinexval["N"]:
        #note index ix,and iy are 1-indexed
        assert val == dsneqsinex.N[ix-1,iy-1].item()



