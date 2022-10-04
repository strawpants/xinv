#!/usr/bin/env python3
## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import unittest
import xarray as xr
import numpy as np
from xinv import *
from xinv.fwd.polynomial import Polynomial


class BuildNeqTest(unittest.TestCase):
    def test_poly(self):
        # xaxis
        xaxis=np.arange(-2,3,0.1)
        
        #initialize the forward operator
        polyfwd=Polynomial(n=2,x=xaxis)
        
        #create polynomial coefficients
        coef=[1.5,3,0.5]
        


        simobs=polyfwd(coef)
        #make some noise!!
        simobs+=np.random.normal(0,0.05,simobs.shape)
        
        # build a normal equation system
        neq=xr.Dataset.xinv.buildneq(polyfwd,simobs)


        #solve the normal equation system
        neq.xinv.solve()
        
        #compare the estimated with the prescribed parameters

        self.assertEqual('foo'.upper(), 'FOO')

    # def test_poly_weighted(self):
        # pass



if __name__ == '__main__':
    unittest.main()

