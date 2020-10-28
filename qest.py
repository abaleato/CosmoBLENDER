import numpy as np
from scipy.interpolate import interp1d
from pyccl.pyutils import _fftlog_transform
from lensing_rec_biases_code import spectra

class experiment:
    def __init__(self, nlev_t, beam_size, lmax, fname_scalar=None, fname_lensed=None, freq_GHz=150.):
        ''' Initialise a cosmology and experimental charactierstics
            - Inputs:
                * nlev_t = temperature noise level, In uK.arcmin.
                * beam_size = beam fwhm (symmetric). In arcmin.
                * lmax = reconstruction lmax.
                * (optional) fname_scalar = CAMB files for unlensed CMB
                * (optiomal) fname_lensed = CAMB files for lensed CMB
        '''
        if fname_scalar is None:
            fname_scalar = '/Users/antonbaleatolizancos/Software/Quicklens-with-fixes/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_scalCls.dat'
        if fname_lensed is None:
            fname_lensed = '/Users/antonbaleatolizancos/Software/Quicklens-with-fixes/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat'

        #Initialise CAMB spectra for filtering
        cl_unl = spectra.get_camb_scalcl(fname_scalar, lmax=lmax)
        cl_len = spectra.get_camb_lensedcl(fname_lensed, lmax=lmax)
        self.clpp = cl_unl.clpp
        self.cltt = cl_len.cltt
        self.ls = cl_len.ls
        self.lmax = lmax
        self.freq_GHz=freq_GHz

        bl = spectra.bl(beam_size, lmax) # beam transfer function.
        self.nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
        
        # Initialise arrays to store the biases
        empty_arr = np.zeros(lmax + 1)
        self.biases = { 'tsz' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr}, 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr}, 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr}}, 'cib' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr}, 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr}, 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr}} }

    def __getattr__(self, spec):
        try:
            return self.biases[spec]
        except KeyError:
            raise AttributeError(spec)

    def get_filtered_profiles_fftlog(self, profile_leg1, profile_leg2=None):
        '''
        Filter the profiles in the way of, e.g., eq. (7.9) of Lewis & Challinor 06.
        Inputs:
            * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
            * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
        Returns:
            * Interpolatable objects from which to get F_1 and F_2 at every multipole.
        '''
        if profile_leg2 is None:
            profile_leg2 = profile_leg1

        F_1_of_l = profile_leg1 / (self.cltt + self.nltt)
        F_2_of_l = self.cltt * profile_leg2/(self.cltt + self.nltt)
            
        al_F_1 = interp1d(self.ls, F_1_of_l, bounds_error=False,  fill_value='extrapolate')
        al_F_2 = interp1d(self.ls, F_2_of_l, bounds_error=False,  fill_value='extrapolate')
        return al_F_1, al_F_2

    def get_unnorm_TT_qe(self, ell_out, profile_leg1, profile_leg2=None, fftlog_way=True, N_l=4*4096, lmin=0.000135, alpha=-1.35):
        '''
        Helper function to get the unnormalised TT QE reconstruction for spherically-symmetric profiles using FFTlog
        Inputs:
            * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
            * (optional) fftlog_way = Bool. Use fftlog if True, Quicklens if False.
            * (optional) N_l = Integer (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
            * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values (e.g., lmin=1e-4) to avoid ringing.
            * (optional) alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
        Returns:
            * If fftlog_way=True, a 1D array containing the unnormalised reconstruction at the multipoles specified in ell_out
        '''
        if fftlog_way:
            al_F_1, al_F_2 = self.get_filtered_profiles_fftlog(profile_leg1, profile_leg2=None)
            return self.unnorm_TT_qe_fftlog(al_F_1, al_F_2, N_l, lmin, alpha)(ell_out)
        else:
            print('Implement QL way!')

    def unnorm_TT_qe_fftlog(self, al_F_1, al_F_2, N_l, lmin, alpha):
        '''
        Compute the unnormalised TT QE reconstruction for spherically symmetric profiles using FFTlog.
        Inputs:
            * al_F_1 = Interpolatable object from which to get F_1 (e.g., in eq. (7.9) of Lewis & Challinor 06) at every multipole.
            * al_F_2 = Interpolatable object from which to get F_2 (e.g., in eq. (7.9) of Lewis & Challinor 06) at every multipole.
            * (optional) N_l = Integer (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
            * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values (e.g., lmin=1e-4) to avoid ringing
            * (optional) alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
        Returns:
            * An interp1d object into which you can plug in an array of ells to get the QE at those ells.
        '''
        ell = np.logspace(np.log10(lmin), np.log10(self.lmax), N_l)

        # The underscore notation _xyz refers to x=hankel order, y=F_y, z=powers of ell
        r_arr_0, f_010 = _fftlog_transform(ell, al_F_1(ell), 2, 0, alpha)
        r_arr_1, f_121 = _fftlog_transform(ell, ell * al_F_2(ell), 2, 1, alpha)
        r_arr_2, f_111 = _fftlog_transform(ell, ell * al_F_1(ell), 2, 1, alpha)
        r_arr_3, f_022 = _fftlog_transform(ell, ell**2 * al_F_2(ell), 2, 0, alpha)
        r_arr_4, f_222 = _fftlog_transform(ell, ell**2 * al_F_2(ell), 2, 2, alpha)

        ell_out_arr, fl_total = _fftlog_transform(r_arr_4, f_121 * (-f_010/r_arr_0 + f_111) + 0.5 * f_010*(-f_022 + f_222) , 2, 0, alpha)
        # Interpolate and correct factors of 2pi from fftlog convetions
        unnormalised_phi = interp1d(ell_out_arr, - (2*np.pi)**3 * fl_total, bounds_error=False, fill_value=0.0)
        return unnormalised_phi
