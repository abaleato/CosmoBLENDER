import numpy as np
from scipy.interpolate import interp1d

def scale_sz(freq=150.):
    ''' f_nu in the literature. this is only the non-relativistic formula. note that the formula in alexs paper is wrong. get it from sehgal et al.'''
    #freq must be in GHz
    freq_hz = freq*1e9
    T_CMB = 2.7255e6
    h = 6.6260755e-27#6.62607004e-34
    K_b = 1.380658e-16#1.38064852e-23
    T_CMB_K = T_CMB/1e6
    x_nu = h*freq_hz/(K_b * T_CMB_K)
    return x_nu * np.cosh(x_nu/2.)/np.sinh(x_nu/2.) - 4

def pkToPell(chi,ks,pk,ellmax=9001):
    # State that the argument in P(k) is P(k*chi), and then set l=k*chi so that P(l/chi)
    interp = interp1d(ks*chi,pk,kind='cubic',bounds_error=False,fill_value=0)
    return interp(np.arange(0,ellmax+1))
