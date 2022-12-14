## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2022 Roelof Rietbroek, rietbroek@utwente.nl

import xarray as xr

import scipy.linalg.blas as blas

@xr.register_dataset_accessor("xinv")
class InverseAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
    
    @staticmethod
    def buildneq(fwd,obs=None,cov=None,apriori=None):
        """Builds a normal equation system from forward operators, data and an accompanying covariance"""
        # if cov is not None:
            #weighted, apply a decorrelation to the observations and forward operator
            # fwd=cov.decorrelate(fwd.jacobian))
            # obs=cov.decorrelate(obs)
        
        # #create a normal matrix and right hand side vector
        A=fwd.jacobian()
        unkcoord=fwd.unknown_coord()
        dsneq=xr.Dataset(coords={unkcoord.name:unkcoord.values})
        if obs is not None:
            dsneq['Atb']=xr.dot(A.T,obs)
        
        # add the symemtric matrix (adds a new dimension without a coord (as it it the same as the unkcoord)
        dsneq['AtA']=([unkcoord.name,unkcoord.name+"_t"],blas.dsyrk(1.0,A.values,trans=1))
        
        dsneq['ltpl']=xr.dot(obs.T,obs)
        return dsneq
   
    def solve(self):
       """Solve the current normal equation system"""

       pass
    # def reduce(self,variables=None):
        # """Reduce (implicitly solve) certain variables in the normal equation system"""
        # return self._obj


    # def fix(self,variables=None):
        # """Fix the indicated variables to its a apriori values"""
        # return self._obj

    # def set_apriori(self,da):
        # """Change the a priori values of the system"""
        # return self._obj


    # def add(self,other):
        # """Add one normal equation system to another. Common parameters will be added, the system will be extended with unique parameters of the other system"""
        # return self._obj
