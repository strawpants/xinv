## Permissions: See the  xinv  license file https://raw.githubusercontent.com/strawpants/xinv/master/LICENSE
## Copyright (c) 2025 Roelof Rietbroek, r.rietbroek@utwente.nl



from xinv.fwd.polynomial import Polynomial
from xinv.fwd.harmonics import SeasonalHarmonics,delta1year
from xinv.fwd.fwdstack import FwdStackOp

def SeasPoly(t0,npoly,x_coord="time",semi_annual=True):
    """
    Convenience function to create a stacked forward operator with a polynomial and seasonal variation
    """

    polyfwd=Polynomial(n=npoly,poly_x=x_coord,cache=True,x0=t0,delta_x=delta1year)
    
    #initialize the stacked forward operator
    fwdstck=FwdStackOp(polyfwd)

    #Append Annual and Seminannual fwd operators
    seasfwd=SeasonalHarmonics(x0=t0,semi_annual=semi_annual,harm_x=x_coord)
    fwdstck.append(seasfwd)
    
    return fwdstck

