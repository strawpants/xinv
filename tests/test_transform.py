import pytest
import xarray as xr
import numpy as np
from xinv import *
import os
from xinv.core.tools import find_ilocs
from xinv.core.attrs import find_xinv_coords,xinv_tp,xinv_st,find_neq_components
from xinv.neq import transform
from xinv.fwd.matrixfwd import MatrixfwdOp
import pandas as pd
from xinv.core.exceptions import XinvIllposedError
from xinv.fwd.polynomial import Polynomial

from fixtures import noisypoly


def test_neq_transform(noisypoly):
    """
    Test the NEQ transform where all parameters are transformed
    """

    #build a base neq system with a polynomial of degree 0 (just a mean)
    x0=noisypoly.attrs['x0']
    deltax=noisypoly.attrs['delta_x']
    npolybase=0
    unkdim='polybase'
    polyfwdbase=Polynomial(n=npolybase,poly_x='x',x0=x0,delta_x=deltax,unknown_dim=unkdim)
    
    #amount of polynomials we would like to resolve after transformation
    npoly_resolve=noisypoly.sizes['poly']-1

    #create a transformation which relates the mean to a polynomial of npoly
    
    std_noise=0.5
    #polynomial forward operator to use in the transformation, this maps a set of higher degree polynomial coefficients (new unknowns) to a mean (previous uknown)
    polyfwd=Polynomial(n=npoly_resolve,poly_x='x',x0=x0,delta_x=deltax,cache=False,obs_dim=unkdim,auxcoords={unkdim:(unkdim,np.arange(npolybase+1))}) #note cache is false because it needs to be recomputed everytime the jacobian is constructed

    neqtrans=None

    # polyfwd2=Polynomial(n=npoly_resolve,poly_x='x',x0=x0,delta_x=deltax,cache=True)
    # dsneq=noisypoly.polyobs.xi.build_normal(polyfwd2,ecov=std_noise*std_noise)
    #solve the joint system
    # dssolcomb=dsneq.xi.solve()
    
    #loop over x slices (make one neq system per x slice and transform and add it)
    for x in noisypoly.x.values:
        daobs=noisypoly.polyobs.sel(x=[x])
        neqsingle=daobs.xi.build_normal(polyfwdbase,ecov=std_noise*std_noise) 
        #transform
        if neqtrans is None:
            neqtrans=neqsingle.xi.transform(polyfwd,x=[x])
        else:
            #add all contributions
            neqtrans=neqtrans.xi.add(neqsingle.xi.transform(polyfwd,x=[x]))
    
    dssol=neqtrans.xi.solve()
    
    #compare with combined system (should give the same results)
    polyfwd2=Polynomial(n=npoly_resolve,poly_x='x',x0=x0,delta_x=deltax,cache=True)
    dsneq=noisypoly.polyobs.xi.build_normal(polyfwd2,ecov=std_noise*std_noise)
    
    #solve the joint system
    dssolcomb=dsneq.xi.solve()
    prenoise=noisypoly.attrs['noise_std']
    tol=3*np.sqrt(np.diag(dssol.COV))*prenoise
    tol=tol.max()
    assert np.allclose(dssol.solution,noisypoly.polytrue,atol=tol)        

    #also check covariance
    assert np.allclose(dssol.COV,dssolcomb.COV)        

def test_neq_partial_transform(noisypoly):
    """
    Test the NEQ transform where some parameters are transformed
    """

    #build a base neq system with a polynomial of degree 0 (just a mean)
    x0=noisypoly.attrs['x0']
    deltax=noisypoly.attrs['delta_x']
    npolybase=0
    unkdim='polybase'
    polyfwdbase=Polynomial(n=npolybase+1,poly_x='x',x0=x0,delta_x=deltax,unknown_dim=unkdim)
    
    #amount of polynomials we would like to resolve after transformation
    npoly_resolve=noisypoly.sizes['poly']-1

    #create a transformation which relates the mean to a polynomial of npoly
    
    std_noise=0.5
    #polynomial forward operator to use in the transformation, this maps a set of higher degree polynomial coefficients (new unknowns) to a mean (previous uknown)
    polyfwd=Polynomial(n=npoly_resolve,poly_x='x',x0=x0,delta_x=deltax,cache=False,obs_dim=unkdim,auxcoords={unkdim:(unkdim,np.arange(npolybase+1))}) #note cache is false because it needs to be recomputed everytime the jacobian is constructed

    neqtrans=None

    
    #loop over x slices (make one neq system per x slice and transform it)
    for x in noisypoly.x.values:
        daobs=noisypoly.polyobs.sel(x=[x])
        neqsingle=daobs.xi.build_normal(polyfwdbase,ecov=std_noise*std_noise) 
        
        #transform
        if neqtrans is None:
            neqtrans=neqsingle.xi.transform(polyfwd,x=[x])
        else:
            #add all contributions
            neqtrans=neqtrans.xi.add(neqsingle.xi.transform(polyfwd,x=[x]))
    
    
    #try solving (should not work because of rank defect
    try:
        dssol=neqtrans.xi.solve()
        #should not get here:
        assert False
    except XinvIllposedError: 
        #expected behavior
        pass


    #select only the new 'poly' group (which should be solvable)
    neqtranspoly=neqtrans.xi.get_group('poly')

    dssol=neqtranspoly.xi.solve()
   
    #compare with combined system (should give the same results)
    polyfwd2=Polynomial(n=npoly_resolve,poly_x='x',x0=x0,delta_x=deltax,cache=True)
    dsneq=noisypoly.polyobs.xi.build_normal(polyfwd2,ecov=std_noise*std_noise)
    #solve the joint system
    dssolcomb=dsneq.xi.solve()
    
    prenoise=noisypoly.attrs['noise_std']
    tol=3*np.sqrt(np.diag(dssol.COV))*prenoise
    tol=tol.max()
    assert np.allclose(dssol.solution,noisypoly.polytrue,atol=tol)        
    
    #also check covariance
    assert np.allclose(dssol.COV,dssolcomb.COV)        
