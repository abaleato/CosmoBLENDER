import numpy as np
from scipy.interpolate import interp1d
import functools
from astropy import units as u
from astropy.cosmology import Planck15

def scale_sz(freq=150.):
    """ f_nu in the literature. this is only the non-relativistic formula. note that the formula in alexs paper is wrong. get it from sehgal et al."""
    #freq must be in GHz
    freq_hz = freq*1e9
    T_CMB = 2.7255e6
    h = 6.6260755e-27#6.62607004e-34
    K_b = 1.380658e-16#1.38064852e-23
    T_CMB_K = T_CMB/1e6
    x_nu = h*freq_hz/(K_b * T_CMB_K)
    return x_nu * np.cosh(x_nu/2.)/np.sinh(x_nu/2.) - 4

def from_Jypersr_to_uK(freq_GHz):
    """ Convert from specific intensity (in Jy sr^-1) to CMB thermodynamic temperature units (in microKelvin),
        assuming infinitely narrow bands.
    - Inputs:
        * freq_GHz = Frequency at which the measurement is made. In units of GHz
    - Returns:
        Multiplicative conversion factor in units of microKelvin/(Jy sr^-1)
    """
    freq = freq_GHz * u.GHz
    equiv = u.thermodynamic_temperature(freq, Planck15.Tcmb0)
    return (1. * u.Jy / u.sr).to(u.uK, equivalencies=equiv)


def split_positive_negative(spectrum):
    """ Separately return the positive and negative parts of an input np.array named spectrum.
    """
    spectrum_pos = spectrum.copy()
    spectrum_neg = spectrum.copy()

    spectrum_pos[spectrum < 0] = np.nan
    spectrum_neg[spectrum > 0] = np.nan
    return spectrum_pos, - spectrum_neg

def pkToPell(chi,ks,pk,ellmax=9001):
    # State that the argument in P(k) is P(k*chi), and then set l=k*chi so that P(l/chi)
    interp = interp1d(ks*chi,pk,kind='cubic',bounds_error=False,fill_value=0)
    return interp(np.arange(0,ellmax+1))

# Now some useful decorators, inspired by James Fergusson

def debug(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if debug_flag:
            arguments = [f"{a}" for a in args]
            karguments = [f"{k}={v}" for k,v in kwargs.items()]
            name = func.__name__
            print("Calling "+name+" with args: "+", ".join(arguments)+" and kwargs: "+", ".join(karguments))
            value = func(*args, **kwargs)
            print("Run function: "+name+", which output: "+repr(value))
            return value
        else:
            return func(*args, **kwargs)
    return wrapper
