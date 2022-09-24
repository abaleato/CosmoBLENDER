import numpy as np
from scipy.interpolate import interp1d
import functools
from astropy import units as u
from astropy.cosmology import Planck15
import quicklens as ql

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
    return (1. * u.Jy / u.sr).to(u.uK, equivalencies=equiv).value


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

def cl2cfft_mod(cl, pix, ls=None, right=0, left=None):
    """
    Adapted from D.Hanson's Quikclens. Returns a maps.cfft object with the pixelization pix,
    with FFT(lx,ly) = linear interpolation of cl[l] at l = sqrt(lx**2 + ly**2).
    """
    ell = pix.get_ell().flatten()
    if ls is None:
        ls = np.arange(0, len(cl))

    ret = ql.maps.cfft(nx=pix.nx, dx=pix.dx,
                       fft=np.array(np.interp(ell, ls, cl, right=right, left=left).reshape(pix.nx, pix.ny),
                                    dtype=complex),
                       ny=pix.ny, dy=pix.dy)

    return ret

def calculate_cl_bias(pix, clee, clpp, lbins, clee_ls=None, clpp_ls=None, aux_e_array=None, aux_p_array=None,
                      left=None):
    '''
    Function to calculate relevant spectra related to lensing B-modes. Adapted from D.Hanson's Quicklens
    '''
    ret = ql.maps.cfft(nx=pix.nx, dx=pix.dx, ny=pix.ny, dy=pix.dy)

    lx, ly = ret.get_lxly()
    l = np.sqrt(lx ** 2 + ly ** 2)
    psi = np.arctan2(lx, -ly)

    clee_array = cl2cfft_mod(clee, ret, ls=clee_ls, left=left).fft
    clpp_array = cl2cfft_mod(clpp, ret, ls=clpp_ls, left=left).fft
    if aux_e_array is not None:
        clee_array = clee_array * aux_e_array
    if aux_p_array is not None:
        clpp_array = clpp_array * aux_p_array

    exp_pos = np.exp(4.j * psi)
    exp_neg = np.exp(-4.j * psi)
    lxsq = lx ** 2
    lysq = ly ** 2
    lxly = lx * ly
    ret.fft = 0.25 * (np.fft.fft2(np.fft.ifft2(2 * clee_array * lxsq) * np.fft.ifft2(clpp_array * lxsq))
                      - np.fft.fft2(
                np.fft.ifft2(clee_array * (exp_pos) * lxsq) * np.fft.ifft2(clpp_array * lxsq)) * exp_neg
                      - np.fft.fft2(
                np.fft.ifft2(clee_array * (exp_neg) * lxsq) * np.fft.ifft2(clpp_array * lxsq)) * exp_pos

                      + np.fft.fft2(np.fft.ifft2(2 * clee_array * lysq) * np.fft.ifft2(clpp_array * lysq))
                      - np.fft.fft2(
                np.fft.ifft2(clee_array * (exp_pos) * lysq) * np.fft.ifft2(clpp_array * lysq)) * exp_neg
                      - np.fft.fft2(
                np.fft.ifft2(clee_array * (exp_neg) * lysq) * np.fft.ifft2(clpp_array * lysq)) * exp_pos

                      + np.fft.fft2(np.fft.ifft2(2 * clee_array * 2 * lxly) * np.fft.ifft2(clpp_array * lxly))
                      - np.fft.fft2(
                np.fft.ifft2(clee_array * (exp_pos) * 2 * lxly) * np.fft.ifft2(clpp_array * lxly)) * exp_neg
                      - np.fft.fft2(
                np.fft.ifft2(clee_array * (exp_neg) * 2 * lxly) * np.fft.ifft2(clpp_array * lxly)) * exp_pos)

    ret *= 1 / (ret.dx * ret.dy)

    return ret.get_ml(lbins)

def gal_window(zs):
    """ Galaxy window function
    """
    # Todo:check this
    return 1./(1.+zs)

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
