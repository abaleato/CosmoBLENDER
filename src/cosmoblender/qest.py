import numpy as np
from scipy.interpolate import interp1d
from pyccl.pyutils import _fftlog_transform
from scipy.integrate import quad
import quicklens as ql
import pickle
from . import tools as tls
import sys
# TODO: install BasicILC
sys.path.insert(0, '/Users/antonbaleatolizancos/Software/BasicILC_py3/')
import cmb_ilc

class experiment:
    def __init__(self, nlev_t=np.array([5.]), beam_size=np.array([1.]), lmax=3500, massCut_Mvir = np.inf, nx=1024,
                 dx_arcmin=1.0, fname_scalar=None, fname_lensed=None, freq_GHz=np.array([150.]), atm_fg=True,
                 MV_ILC_bool=False, deproject_tSZ=False, deproject_CIB=False, bare_bones=False):
        """ Initialise a cosmology and experimental charactierstics
            - Inputs:
                * nlev_t = np array. Temperature noise level, in uK.arcmin. Either single value or one for each freq
                * beam_size = np array. beam fwhm (symmetric). In arcmin. Either single value or one for each freq
                * lmax = reconstruction lmax.
                * (optional) massCut_Mvir = Maximum halo virial masss, in solar masses. Default is no cut (infinite)
                * (optional) fname_scalar = CAMB files for unlensed CMB
                * (optional) fname_lensed = CAMB files for lensed CMB
                * (otional) nx = int. Width in number of pixels of grid used in quicklens computations
                * (optional) dx = float. Pixel width in arcmin for quicklens computations
                * (optional) freq_GHz =np array of one or many floats. Frequency of observqtion (in GHZ). If array,
                                        frequencies that get combined as ILC using ILC_weights as weights
                * (optional) atm_fg = Whether or not to include atmospheric fg power in inverse-variance filter
                * (optional) MV_ILC_bool = Bool. If true, form a MV ILC of freqs
                * (optional) deproject_tSZ = Bool. If true, form ILC deprojecting tSZ and retaining unit response to CMB
                * (optional) deproject_CIB = Bool. If true, form ILC deprojecting CIB and retaining unit response to CMB
                * (optional) bare_bones= Bool. If True, don't run any of the costly operations at initialisation
        """
        if fname_scalar is None:
            fname_scalar = None#'~/Software/Quicklens-with-fixes/quicklens/data\/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_scalCls.dat'
        if fname_lensed is None:
            fname_lensed = None#'~/Software/Quicklens-with-fixes/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat'

        #Initialise CAMB spectra for filtering
        self.cl_unl = ql.spec.get_camb_scalcl(fname_scalar, lmax=lmax)
        self.cl_len = ql.spec.get_camb_lensedcl(fname_lensed, lmax=lmax)
        self.ls = self.cl_len.ls
        self.lmax = lmax
        self.lmin = 2
        self.freq_GHz = freq_GHz

        self.massCut = massCut_Mvir #Convert from M_vir (which is what Alex uses) to M_200 (which is what the
                                    # Tinker mass function in hmvec uses) using the relation from White 01.

        self.nlev_t = nlev_t
        self.nlev_p = np.sqrt(2) * nlev_t
        self.beam_size = beam_size

        # Set up grid for Quicklens calculations
        self.nx = nx
        self.dx = dx_arcmin/60./180.*np.pi # pixel width in radians.
        self.pix = ql.maps.cfft(self.nx, self.dx)

        #TODO: Calculate W_E in the multifrequency case
        #self.nlee = (np.pi / 180. / 60. * self.nlev_p) ** 2 / self.bl ** 2
        #self.W_E = np.nan_to_num(self.cl_len.clee / (self.cl_len.clee + self.nlee))

        self.MV_ILC_bool = MV_ILC_bool
        self.deproject_tSZ = deproject_tSZ
        self.deproject_CIB = deproject_CIB
        if not bare_bones:
            #Initialise sky model
            self.sky = cmb_ilc.CMBILC(freq_GHz*1e9, beam_size, nlev_t, atm=atm_fg, lMaxT=self.lmax)
            if len(self.freq_GHz)>1:
                # In cases where there are several, compute ILC weights for combining different channels
                assert MV_ILC_bool or deproject_tSZ or deproject_CIB, 'Please indicate how to combine different channels'
                assert not (MV_ILC_bool and (deproject_tSZ or deproject_CIB)), 'Only one ILC type at a time!'
                self.get_ilc_weights()
            # Compute total TT power (incl. noise, fgs, cmb) for use in inverse-variance filtering
            self.get_total_TT_power()

            # Calculate inverse-variance filters
            self.inverse_variance_filters()
            # Calculate QE norm
            self.get_qe_norm()
            self.nlpp = self.get_nlpp()
            self.W_phi = self.cl_unl.clpp / (self.cl_unl.clpp + self.nlpp)

        # Initialise an empty dictionary to store the biases
        empty_arr = {}
        self.biases = { 'ells': empty_arr,
                        'second_bispec_bias_ells': empty_arr,
                        'tsz' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'cross_w_gals' : {'1h' : empty_arr, '2h' : empty_arr}},
                        'cib' : {'trispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'prim_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'second_bispec' : {'1h' : empty_arr, '2h' : empty_arr},
                                 'cross_w_gals' : {'1h' : empty_arr, '2h' : empty_arr}},
                        'mixed': {'trispec': {'1h': empty_arr, '2h': empty_arr},
                                'prim_bispec': {'1h': empty_arr, '2h': empty_arr},
                                'second_bispec': {'1h': empty_arr, '2h': empty_arr},
                                 'cross_w_gals' : {'1h' : empty_arr, '2h' : empty_arr}} }

    def inverse_variance_filters(self):
        """
        Calculate the inverse-variance filters to be applied to the fields prior to lensing reconstruction
        """
        lmin = 2 #TODO: define this as a method of the class
        # Initialize some dummy object that are required by quicklens
        # Initialise a dummy set of maps for the computation
        tmap = qmap = umap = np.random.randn(self.nx, self.nx)
        tqumap = ql.maps.tqumap(self.nx, self.dx, maps=[tmap, qmap, umap])
        transf = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : self.lmax, 'cltt' : np.ones(self.lmax+1),
                                                     'clee' : np.ones(self.lmax+1),
                                                     'clbb' : np.ones(self.lmax+1)} ), self.pix)
        cl_tot_theory  = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : self.lmax, 'cltt' : self.cltt_tot,
                                                          'clee' : np.zeros(self.lmax+1),
                                                              'clbb' :np.zeros(self.lmax+1)} ) )
        # TODO: find a neater way of doing this using ivf.library_diag()
        self.ivf_lib = ql.sims.ivf.library_l_mask(ql.sims.ivf.library_diag_emp(tqumap, cl_tot_theory, transf=transf,
                                                                               nlev_t=0, nlev_p=0),
                                                  lmin=lmin, lmax=self.lmax)

    def get_ilc_weights(self):
        """
        Get the harmonic ILC weights
        """
        lmin_cutoff = 14
        ell_spacing = 100 # Sum of weights is still 1 to 1 part in 10^14 even with ell_spacing=100
        # Evaluate only at discrete ells, and interpolate later.
        W_sILC_Ls = np.arange(lmin_cutoff, self.lmax, ell_spacing)

        if self.MV_ILC_bool:
            W_sILC = np.array(
                list(map(self.sky.weightsIlcCmb, W_sILC_Ls)))
        elif self.deproject_tSZ and self.deproject_CIB:
            W_sILC = np.array(
                list(map(self.sky.weightsDeprojTszCIB, W_sILC_Ls)))
        elif self.deproject_tSZ:
            W_sILC = np.array(
                list(map(self.sky.weightsDeprojTsz, W_sILC_Ls)))
        elif self.deproject_CIB:
            W_sILC = np.array(
                list(map(self.sky.weightsDeprojCIB, W_sILC_Ls)))

        self.ILC_weights, self.ILC_weights_ells = tls.spline_interpolate_weights(W_sILC, W_sILC_Ls, self.lmax)
        return

    def get_tsz_filter(self):
        """
        Calculate the ell-dependent filter to be applied to the y-profile harmonics. In the single-frequency scenario,
        this just applies the tSZ frequency-dependence. If doing ILC cleaning, it includes both the frequency
        dependence and the effect of the frequency-and-ell-dependent weights
        """
        if len(self.freq_GHz)>1:
            # Multiply ILC weights at each freq by tSZ scaling at that freq, then sum them together at every multipole
            tsz_filter = np.sum(tls.scale_sz(self.freq_GHz) * self.ILC_weights, axis=1)
            # Return the filter interpolated at every ell where we will perform lensing recs, i.e. [0, self.lmax]
            #TODO: I don't think this interpolation step is needed anymore
            return np.interp(np.arange(self.lmax+1), self.ILC_weights_ells, tsz_filter, left=0, right=0)
        else:
            # Single-frequency scenario. Return a single number.
            return tls.scale_sz(self.freq_GHz)

    def get_total_TT_power(self):
        """
        Get total TT power from CMB, noise and fgs.
        Note that if both self.deproject_tSZ=1 and self.deproject_CIB=1, both are deprojected
        """
        #TODO: Why can't we get ells below 10 in cltt_tot?
        if len(self.freq_GHz)==1:
            self.cltt_tot = self.sky.cmb[0, 0].ftotalTT(self.cl_unl.ls)
        else:
            nL = 201
            L = np.logspace(np.log10(self.lmin), np.log10(self.lmax), nL)
            # ToDo: sample better in L
            if self.MV_ILC_bool:
                f = lambda l: self.sky.powerIlc(self.sky.weightsIlcCmb(l), l)
            elif self.deproject_tSZ and self.deproject_CIB:
                f = lambda l: self.sky.powerIlc(self.sky.weightsDeprojTszCIB(l), l)
            elif self.deproject_tSZ:
                f = lambda l: self.sky.powerIlc(self.sky.weightsDeprojTsz(l), l)
            elif self.deproject_CIB:
                f = lambda l: self.sky.powerIlc(self.sky.weightsDeprojCIB(l), l)
            #TODO: turn zeros into infinities to avoid issues when dividing by this
            self.cltt_tot = np.interp(self.cl_unl.ls, L, np.array(list(map(f, L))))
        # Avoid infinities when dividing by inverse variance
        self.cltt_tot[np.where(np.isnan(self.cltt_tot))] = np.inf

    def get_qe_norm(self, key='ptt'):
        """
        Calculate the QE normalisation as the reciprocal of the N^{(0)} bias
        Inputs:
            * (optional) key = String. The quadratic estimator key. Default is 'ptt' for TT
        """
        self.qest_lib = ql.sims.qest.library(self.cl_unl, self.cl_len, self.ivf_lib)
        self.qe_norm = self.qest_lib.get_qr(key)

    def get_nlpp(self):
        # TODO: this N0 is not smooth. Find a better way to calculate
        lbins = np.arange(8, 3000, 30)
        norm = self.qe_norm.get_ml(lbins)
        return np.interp(self.cl_unl.ls, norm.ls, np.nan_to_num(1./norm.specs['cl']))

    def __getattr__(self, spec):
        try:
            return self.biases[spec]
        except KeyError:
            raise AttributeError(spec)

    def __str__(self):
        """ Print out halo model calculator properties """
        massCut = '{:.2e}'.format(self.massCut)
        return 'Mass Cut: ' + str(massCut) + '  lmax: ' + str(self.lmax) + '  Beam FWHM: '+ str(self.beam_size) + \
               ' Noise (uK arcmin): ' + str(self.nlev_t) + '  Freq (GHz): ' + str(self.freq_GHz)

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

        F_1_of_l = np.nan_to_num(profile_leg1 / self.cltt_tot)
        F_2_of_l = np.nan_to_num(self.cl_len.cltt * profile_leg2/ self.cltt_tot)

        al_F_1 = interp1d(self.ls, F_1_of_l, bounds_error=False,  fill_value='extrapolate')
        al_F_2 = interp1d(self.ls, F_2_of_l, bounds_error=False,  fill_value='extrapolate')
        return al_F_1, al_F_2

    def unnorm_TT_qe_fftlog(self, al_F_1, al_F_2, N_l, lmin, alpha):
        """
        Compute the unnormalised TT QE reconstruction for spherically symmetric profiles using FFTlog.
        Inputs:
            * al_F_1 = Interpolatable object from which to get F_1 (e.g., in eq. (7.9) of Lewis & Challinor 06)
                       at every multipole.
            * al_F_2 = Interpolatable object from which to get F_2 (e.g., in eq. (7.9) of Lewis & Challinor 06)
                       at every multipole.
            * (optional) N_l = Int (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
            * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values
                                (e.g., lmin=1e-4) to avoid ringing
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

        ell_out_arr, fl_total = _fftlog_transform(r_arr_4, f_121 * (-f_010/r_arr_0 + f_111)
                                                  + 0.5 * f_010*(-f_022 + f_222) , 2, 0, alpha)
        # Interpolate and correct factors of 2pi from fftlog conventions
        unnormalised_phi = interp1d(ell_out_arr, - (2*np.pi)**3 * fl_total, bounds_error=False, fill_value=0.0)
        return unnormalised_phi

    def get_TT_qe(self, fftlog_way, ell_out, profile_leg1, profile_leg2=None, N_l=2*4096, lmin=0.000135, alpha=-1.35,
                  norm_bin_width=40, key='ptt'):
        """
        Helper function to get the TT QE reconstruction for spherically-symmetric profiles using FFTlog
        Inputs:
            * fftlog_way = Bool. If true, use fftlog reconstruction. Otherwise use quicklens.
            * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
            * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
            * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
            * (optional) N_l = Integer (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
            * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values
                                (e.g., lmin=1e-4) to avoid ringing.
            * (optional) alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
            * (optional) norm_bin_width = int. Bin width to use when taking spectra of the semi-analytic QE
                                          normalisation (for fftlog only)
        Returns:
            * If fftlog_way=True, a 1D array containing the unnormalised reconstruction at the multipoles specified in ell_out
            * (optional) key = String. The quadratic estimator key for quicklens. Default is 'ptt' for TT

        """
        if profile_leg2 is None:
            profile_leg2 = profile_leg1
        if fftlog_way:
            al_F_1, al_F_2 = self.get_filtered_profiles_fftlog(profile_leg1, profile_leg2)
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
            unnormalized_phi = self.qest_lib.get_qft(key, tft1, 0*tft1.copy(), 0*tft1.copy(),
                                                     tft2, 0*tft1.copy(), 0*tft1.copy())
            # In QL, the unnormalised reconstruction (obtained via eval_flatsky()) comes with a factor of sqrt(skyarea)
            A_sky = (self.dx*self.nx)**2
            #Normalize the reconstruction
            return np.nan_to_num(unnormalized_phi.fft[:,:] / self.qe_norm.fft[:,:]) /np.sqrt(A_sky)

    def get_brute_force_unnorm_TT_qe(self, ell_out, profile_leg1, profile_leg2=None):
        """
        Slow but sure method to calculate the 1D TT QE reconstruction.
        Scales as O(N^3), but useful as a cross-check of get_unnorm_TT_qe(fftlog_way=True)
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