import numpy as np
from scipy.interpolate import interp1d
from pyccl.pyutils import _fftlog_transform
from scipy.integrate import quad
import quicklens as ql
import pickle
import sys

class experiment:
    def __init__(self, nlev_t, beam_size, lmax, massCut_Mvir = np.inf, nx=512, dx_arcmin=1.0, fname_scalar=None, fname_lensed=None, freq_GHz=150.):
        """ Initialise a cosmology and experimental charactierstics
            - Inputs:
                * nlev_t = temperature noise level, In uK.arcmin.
                * beam_size = beam fwhm (symmetric). In arcmin.
                * lmax = reconstruction lmax.
                * (optional) massCut_Mvir = Maximum halo virial masss, in solar masses. Default is no cut (infinite)
                * (optional) fname_scalar = CAMB files for unlensed CMB
                * (optional) fname_lensed = CAMB files for lensed CMB
                * (otional) nx = int. Width in number of pixels of grid used in quicklens computations
                * (optional) dx = float. Pixel width in arcmin for quicklens computations
        """
        if fname_scalar is None:
            fname_scalar = None#'~/Software/Quicklens-with-fixes/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_scalCls.dat'
        if fname_lensed is None:
            fname_lensed = None#'~/Software/Quicklens-with-fixes/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat'

        #Initialise CAMB spectra for filtering
        self.cl_unl = ql.spec.get_camb_scalcl(fname_scalar, lmax=lmax)
        self.cl_len = ql.spec.get_camb_lensedcl(fname_lensed, lmax=lmax)
        self.ls = self.cl_len.ls
        self.lmax = lmax
        self.freq_GHz=freq_GHz
        self.massCut = massCut_Mvir #Convert from M_vir (which is what Alex uses) to M_200 (which is what the
                                    # Tinker mass function in hmvec uses) using the relation from White 01.

        self.nlev_t = nlev_t
        self.beam_size = beam_size
        self.bl = ql.spec.bl(beam_size, lmax) # beam transfer function.
        self.nltt = (np.pi/180./60.*nlev_t)**2 / self.bl**2

        # Set up grid for Quicklens calculations
        self.nx = nx
        self.dx = dx_arcmin/60./180.*np.pi # pixel width in radians.
        self.pix = ql.maps.cfft(self.nx, self.dx)

        # Calculate inverse-variance filters
        self.inverse_variance_filters()
        # Calculate QE norm
        self.get_qe_norm()

        # Initialise an empty dictionary to store the biases
        empty_arr = {}
        self.biases = { 'ells': empty_arr,
                        'second_bispec_bias_ells': empty_arr,
                        'tsz' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr}},
                        'cib' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr}},
                        'mixed': {'trispec': {'1h': empty_arr, '2h': empty_arr},
                                'prim_bispec': {'1h': empty_arr, '2h': empty_arr},
                                'second_bispec': {'1h': empty_arr, '2h': empty_arr}} }

    def inverse_variance_filters(self):
        """
        Calculate the inverse-variance filters to be applied to the fields prior to lensing reconstruction
        """
        lmin = 2
        # Initialise a dummy set of maps for the computation
        tmap = qmap = umap = np.random.randn(self.nx,self.nx)
        tqumap = ql.maps.tqumap(self.nx, self.dx, maps=[tmap,qmap,umap])
        #Define the beam transfer function and pixelisation window function.
        #This will be used for beam deconvolution and inverse-variance filtering
        transf = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : self.lmax, 'cltt' : self.bl, 'clee' : self.bl, 'clbb' : self.bl} ), self.pix)
        cl_theory  = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : self.lmax, 'cltt' : self.cl_len.cltt, 'clee' : self.cl_len.clee, 'clbb' : self.cl_len.clbb} ) )
        self.ivf_lib = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag_emp(tqumap, cl_theory, transf=transf, nlev_t=self.nlev_t), lmin=lmin, lmax=self.lmax)

    def get_qe_norm(self, key='ptt'):
        """
        Calculate the QE normalisation as the reciprocal of the N^{(0)} bias
        Inputs:
            * (optional) key = String. The quadratic estimator key. Default is 'ptt' for TT
        """
        self.qest_lib = ql.sims.qest.library(self.cl_unl, self.cl_len, self.ivf_lib)
        self.qe_norm = self.qest_lib.get_qr(key)

    def __getattr__(self, spec):
        try:
            return self.biases[spec]
        except KeyError:
            raise AttributeError(spec)

    def __str__(self):
        """ Print out halo model calculator properties """
        massCut = '{:.2e}'.format(self.massCut)
        beam_size = '{:.2f}'.format(self.beam_size)
        nlev_t = '{:.2f}'.format(self.nlev_t)
        freq_GHz = '{:.2f}'.format(self.freq_GHz)

        return 'Mass Cut: ' + massCut + '  lmax: ' + str(self.lmax) + '  Beam FWHM: '+ beam_size + ' Noise (uK arcmin): ' + nlev_t + '  Freq (GHz): ' + freq_GHz

    def save_biases(self, output_filename='./dict_with_biases'):
        """
        Save the dictionary of biases to file
        Inputs:
            * output_filename = str. Output filename
        """
        with open(output_filename+'.pkl', 'wb') as output:
            pickle.dump(self.biases, output, pickle.HIGHEST_PROTOCOL)

    def get_filtered_profiles_fftlog(self, profile_leg1, profile_leg2=None):
        """
        Filter the profiles in the way of, e.g., eq. (7.9) of Lewis & Challinor 06.
        Inputs:
            * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
            * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
        Returns:
            * Interpolatable objects from which to get F_1 and F_2 at every multipole.
        """
        if profile_leg2 is None:
            profile_leg2 = profile_leg1

        F_1_of_l = profile_leg1 / (self.cl_len.cltt + self.nltt)
        F_2_of_l = self.cl_len.cltt * profile_leg2/(self.cl_len.cltt + self.nltt)
            
        al_F_1 = interp1d(self.ls, F_1_of_l, bounds_error=False,  fill_value='extrapolate')
        al_F_2 = interp1d(self.ls, F_2_of_l, bounds_error=False,  fill_value='extrapolate')
        return al_F_1, al_F_2

    def unnorm_TT_qe_fftlog(self, al_F_1, al_F_2, N_l, lmin, alpha):
        """
        Compute the unnormalised TT QE reconstruction for spherically symmetric profiles using FFTlog.
        Inputs:
            * al_F_1 = Interpolatable object from which to get F_1 (e.g., in eq. (7.9) of Lewis & Challinor 06) at every multipole.
            * al_F_2 = Interpolatable object from which to get F_2 (e.g., in eq. (7.9) of Lewis & Challinor 06) at every multipole.
            * (optional) N_l = Integer (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
            * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values (e.g., lmin=1e-4) to avoid ringing
            * (optional) alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
        Returns:
            * An interp1d object into which you can plug in an array of ells to get the QE at those ells.
        """
        ell = np.logspace(np.log10(lmin), np.log10(self.lmax), N_l)

        # The underscore notation _xyz refers to x=hankel order, y=F_y, z=powers of ell
        r_arr_0, f_010 = _fftlog_transform(ell, al_F_1(ell), 2, 0, alpha)
        r_arr_1, f_121 = _fftlog_transform(ell, ell * al_F_2(ell), 2, 1, alpha)
        r_arr_2, f_111 = _fftlog_transform(ell, ell * al_F_1(ell), 2, 1, alpha)
        r_arr_3, f_022 = _fftlog_transform(ell, ell**2 * al_F_2(ell), 2, 0, alpha)
        r_arr_4, f_222 = _fftlog_transform(ell, ell**2 * al_F_2(ell), 2, 2, alpha)

        ell_out_arr, fl_total = _fftlog_transform(r_arr_4, f_121 * (-f_010/r_arr_0 + f_111) + 0.5 * f_010*(-f_022 + f_222) , 2, 0, alpha)
        # Interpolate and correct factors of 2pi from fftlog conventions
        unnormalised_phi = interp1d(ell_out_arr, - (2*np.pi)**3 * fl_total, bounds_error=False, fill_value=0.0)
        return unnormalised_phi

    def get_TT_qe(self, fftlog_way, ell_out, profile_leg1, profile_leg2=None, N_l=2*4096, lmin=0.000135, alpha=-1.35, norm_bin_width=40, key='ptt'):
        """
        Helper function to get the TT QE reconstruction for spherically-symmetric profiles using FFTlog
        Inputs:
            * fftlog_way = Bool. If true, use fftlog reconstruction. Otherwise use quicklens.
            * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
            * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
            * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
            * (optional) N_l = Integer (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
            * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values (e.g., lmin=1e-4) to avoid ringing.
            * (optional) alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
            * (optional) norm_bin_width = int. Bin width to use when taking spectra of the semi-analytic QE normalisation (for fftlog only)
        Returns:
            * If fftlog_way=True, a 1D array containing the unnormalised reconstruction at the multipoles specified in ell_out
            * (optional) key = String. The quadratic estimator key for quicklens. Default is 'ptt' for TT

        """
        if fftlog_way:
            al_F_1, al_F_2 = self.get_filtered_profiles_fftlog(profile_leg1, profile_leg2=None)
            # Calculate unnormalised QE
            unnorm_TT_qe = self.unnorm_TT_qe_fftlog(al_F_1, al_F_2, N_l, lmin, alpha)(ell_out)
            # Project the QE normalisation to 1D
            lbins = np.arange(lmin, self.lmax, norm_bin_width)
            qe_norm_1D = self.qe_norm.get_ml(lbins)

            # Apply a convention correction to match Quicklens
            conv_corr = 1/(2*np.pi)
            return conv_corr * np.nan_to_num( unnorm_TT_qe / np.interp(ell_out, qe_norm_1D.ls, qe_norm_1D.specs['cl']) )
        else:
            tft1 = ql.spec.cl2cfft(profile_leg1, self.pix)

            # Apply filters and do lensing reconstruction
            t_filter = self.ivf_lib.get_fl().get_cffts()[0]
            tft1.fft *=t_filter.fft
            if profile_leg2 is None:
                tft2 = tft1.copy()
            else:
                tft2 = ql.spec.cl2cfft(profile_leg2, self.pix)
                tft2.fft *= t_filter.fft
            unnormalized_phi = self.qest_lib.get_qft(key, tft1, 0*tft1.copy(), 0*tft1.copy(), tft2, 0*tft1.copy(), 0*tft1.copy())
            # In QL, the unnormalised reconstruction (obtained via eval_flatsky()) comes with a factor of sqrt(skyarea)
            A_sky = (self.dx*self.nx)**2
            #Normalize the reconstruction
            return np.nan_to_num(unnormalized_phi.fft[:,:] / self.qe_norm.fft[:,:]) /np.sqrt(A_sky)

    def get_brute_force_unnorm_TT_qe(self, ell_out, profile_leg1, profile_leg2=None):
        """
        Slow but sure method to calculate the 1D TT QE reconstruction. Scales as O(N^3), but useful as a cross-check of get_unnorm_TT_qe(fftlog_way=True)
        Inputs:
            * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
            * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
            * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
        """
        def ell_dependence(L, l, lp):
            '''L is outter multipole'''
            if (L+l>=lp) and (L+lp>=l) and (l+lp>=L):
                #check triangle inequality
                if (L+l==lp) or (L+lp==l) or (l+lp==L):
                    # integrand is singular at the triangle equality
                    print('dealing with integrable singularity by setting to 0')
                    return 0
                return 2 * ( (L**2 + lp**2 - l**2) / (2*L*lp) )* ( 1 - ((L**2 + lp**2 - l**2) / (2*L*lp) )**2  )**(-0.5)
            else:
                return 0

        def inner_integrand(lp, L, l):
            return lp * al_F_2(lp) * ell_dependence(L, l, lp)

        def outer_integrand(l, L):
            return l * al_F_1(l) * quad(inner_integrand, 1, self.lmax, args=(L, l))[0]

        al_F_1, al_F_2 = self.get_filtered_profiles_fftlog(profile_leg1, profile_leg2=None)
        output_unnormalised_phi = np.zeros(ell_out.shape)
        for i,L in enumerate(ell_out):
            output_unnormalised_phi[i] = quad(outer_integrand, 1, self.lmax, args=L)[0]/(2*np.pi)

        return output_unnormalised_phi

def load_dict_of_biases(filename='./dict_with_biases.pkl'):
    """
    Load a dictionary of biases that was previously saved using experiment.save_biases()
    Inputs:
        * filename = str. Filename for the pickle object to be loaded
    Returns:
        * Dict of biases with indexing as in experiment.biases
    """
    with open(filename, 'rb') as input:
        experiment_object = pickle.load(input)
    print('Successfully loaded experiment object with properties:\n')
    print(experiment_object)
    return experiment_object