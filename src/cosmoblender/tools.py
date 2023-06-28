import numpy as np
from scipy.interpolate import interp1d
import functools
from astropy import units as u
from astropy.cosmology import Planck18
import quicklens as ql
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import CubicSpline

def scale_sz(freq=150.):
    """
    f_nu in the literature. This is only the non-relativistic formula.
    Note that the formula in alexs paper is wrong. get it from sehgal et al.
    """
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
    equiv = u.thermodynamic_temperature(freq, Planck18.Tcmb0)
    return (1. * u.Jy / u.sr).to(u.uK, equivalencies=equiv).value

def spline_interpolate_weights(orig_weights, orig_ells, lmax):
    '''
    Enlarge the array of ILC weights, interpolating inbetween the multipoles sampled
    Inputs:
        - orig_weights = (nells,nchannels) np array of ILC weights
        - orig_ells = (nells,) np array where the weights are defined
        - lmax = int. lmax of the output arrays
    Returns:
        - new_weights = (lmax+1,nchannels) np array. interpolated ILC weights
        - new_ells = (lmax+1,) np array. Multipoles where new_weights is defined
    '''
    new_weights = np.ones((lmax+1, orig_weights.shape[1]))
    new_ells = np.arange(lmax+1)

    for row in range(new_weights.shape[1]):
        cs = CubicSpline(orig_ells, orig_weights[:,row], extrapolate=False)
        new_weights[:,row] = cs(new_ells)
    return new_weights, new_ells

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
    interp = interp1d(ks*chi - 0.5,pk,kind='cubic',bounds_error=False,fill_value=0)
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

def gal_window(zs, gzs, gdndz=None):
    """ Galaxy window function
    zs = zs at which we want to evaluate the dndz
    gzs = zs at which dndz is defined
    gdndz = H(z)*dndz
    """
    gzs = np.array(gzs).reshape(-1)
    if gzs.size > 1:
        nznorm = np.trapz(gdndz, gzs)
        Wz2s = gdndz / nznorm
    else:
        Wz2s = np.ones_like(gzs)
    return np.interp(zs, gzs, Wz2s, left=0, right=0)

def y_window(hcos):
    """ Window function for tSZ projection. In theory this is just W(chi)=a(chi) (see Appendix A of Hill & Pajer 13)
       However, here we need to adapt to how things are implemented in hmvec: the a(z) is absorbed into the projected
       profile, which in addition y_3D_hmvec = y_3D_theory /H(chi)
    """
    return hcos.h_of_z(hcos.zs)

def my_lensing_window(hcos, zs, dndz=None):
    """
    Wrapper around hmvec's lensing_window to have the dependence on H(z) that is more common in the literature
    """
    return hcos.lensing_window(hcos.zs,zs,dndz)

def limber_itgrnd_kernel(hcos, polyspectrum_order):
    '''
    Prefactor in Limber integrals of nth-order polyspectra: chi**(2(1-n))
    Assumes window function have absorbed any dchi/dz
    '''
    return hcos.h_of_z(hcos.zs) * hcos.comoving_radial_distance(hcos.zs) ** (2 * (1 - polyspectrum_order))

# Functions associated with galaxy hods

def get_DESI_surface_ngal_of_z(sample, n_of_z_dir='/Users/antonbaleatolizancos/Projects/kappa_delensing/DESI_dndzs/'):
    # TODO: Make this usable by anyone.
    ''' Return a numpy array with the galaxy number density per unit area in the sky for various DESI samples,
        as given in https://desi.lbl.gov/trac/wiki/keyprojects/y1kp1#no1
        Inputs:
            -  sample = string. One of {'bgs','lrg','bgs'}
        Returns:
            - z_mean = Float. Central redshift of each bin of the given n(z)
            - surface_ngal_of_z = binned ngal(z) in units of deg^{-2}
    '''
    table = np.loadtxt(n_of_z_dir + 'nz_{}_final.dat'.format(sample), skiprows=1)
    zmin = table[:, 0]
    zmax = table[:, 1]
    surface_ngal_of_z = table[:, 2]
    return (zmax + zmin) / 2., surface_ngal_of_z


def comoving_density_single_bin(ntot, z_low, z_hi, area, H0=68., Om0=0.3, error=False):
    '''
    Calculate co-moving number density. From Rongpu Zhou

    Inputs
    ------
    ntot: sample size;
    z_low, z_hi: lower and upper limit of the redshift bin;
    area: area in sq. deg.

    Output
    ------
    density: comoving number density in units of per Mpc^3;
    density_err (optional): Poisson error on density.
    '''

    area_sphere = 41252.96  # Total area of the sky in sq. deg.
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    volume = area / area_sphere * (cosmo.comoving_volume(z_hi) - cosmo.comoving_volume(z_low))
    # convert to Python scalar
    volume = float(volume / (1 * u.Mpc) ** 3)
    density = ntot / volume
    if error:
        density_err = np.sqrt(ntot) / volume
        return density, density_err
    else:
        return density


def get_comoving_from_surface_ngal(z_bins, surface_ngal):
    '''
    Convert a surface number density of galaxies (at each z) into its comoving number density (at each z)
    Inputs:
        - z_bins = Array of size (nzs,). Central redshift of each bin. We assume the bins are uniform
        - surface_ngal = Array of size (nzs,). Surface number density (in deg^{-2} units) at each of z_bins
        Returns:
        - comoving_ngal = Array of size (nzs,). Comoving number density (in Mpc^{-3} units) at each of z_bins
    '''
    comoving_ngal = np.zeros_like(surface_ngal)

    # Get bin width, assuming bins are uniform
    Delta_z = (z_bins[1] - z_bins[0])
    for i, z_bin in enumerate(z_bins):
        # TODO: vectorize this
        comoving_ngal[i] = comoving_density_single_bin(surface_ngal[i], z_bin - Delta_z / 2., z_bin + Delta_z / 2., 1.)
    return comoving_ngal


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
