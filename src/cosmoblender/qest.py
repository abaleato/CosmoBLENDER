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
import concurrent
from scipy.special import roots_legendre

class Exp_minimal:
    """ A helper class to encapsulate some essential attributes of experiment() objects to be passed to parallelized
        workers, saving as much memory as possible
        - Inputs:
            * exp = an instance of the experiment() class
    """
    def __init__(self, exp):
        dict = {"cltt_tot": exp.cltt_tot, "qe_norm_at_lbins_sec_bispec": exp.qe_norm_at_lbins_sec_bispec,
                "lmax": exp.lmax, "nx": exp.nx, "dx": exp.dx, "pix": exp.pix, "tsz_filter": exp.tsz_filter,
                "massCut": exp.massCut, "ls":exp.ls, "cl_len":exp.cl_len, "cl_unl":exp.cl_unl, "qest_lib":exp.qest_lib,
                "ivf_lib":exp.ivf_lib, "qe_norm":exp.qe_norm_compressed, "nx_secbispec":exp.nx_secbispec,
                "dx_secbispec":exp.dx_secbispec, "weights_mat_total":exp.weights_mat_total, "nodes":exp.nodes}
        self.__dict__ = dict

class experiment:
    def __init__(self, nlev_t=np.array([5.]), beam_size=np.array([1.]), lmax=3500, massCut_Mvir = np.inf, nx=1024,
                 dx_arcmin=1., nx_secbispec=256, dx_arcmin_secbispec=1.0, fname_scalar=None, fname_lensed=None, freq_GHz=np.array([150.]), fg=True, atm_fg=True,
                 MV_ILC_bool=False, deproject_tSZ=False, deproject_CIB=False, bare_bones=False, nlee=None,
                 gauss_order=1000):
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
                * (otional) nx_secbispec = int. Same as nx, but for secondary bispectrum bias calculation
                * (optional) dx_arcmin_secbispec = float. Same as dx, but for secondary bispectrum bias calculation
                * (optional) freq_GHz =np array of one or many floats. Frequency of observqtion (in GHZ). If array,
                                        frequencies that get combined as ILC using ILC_weights as weights
                * (optional) fg = Whether or not to include non-atmospheric fg power in inverse-variance filter
                * (optional) atm_fg = Whether or not to include atmospheric fg power in inverse-variance filter
                * (optional) MV_ILC_bool = Bool. If true, form a MV ILC of freqs
                * (optional) deproject_tSZ = Bool. If true, form ILC deprojecting tSZ and retaining unit response to CMB
                * (optional) deproject_CIB = Bool. If true, form ILC deprojecting CIB and retaining unit response to CMB
                * (optional) bare_bones= Bool. If True, don't run any of the costly operations at initialisation
                * (optional) nlee = np array of size lmax+1 containing E-mode noise power for delensing template
                * (optional) gauss_order= int. Order of the Gaussian quadrature used to compute analytic QE
        """
        if fname_scalar is None:
            fname_scalar = None#'~/Software/Quicklens-with-fixes/quicklens/data\/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_scalCls.dat'
        if fname_lensed is None:
            fname_lensed = None#'~/Software/Quicklens-with-fixes/quicklens/data/cl/planck_wp_highL/planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat'

        #Initialise CAMB spectra for filtering
        self.cl_unl = ql.spec.get_camb_scalcl(fname_scalar, lmax=lmax)
        self.cl_len = ql.spec.get_camb_lensedcl(fname_lensed, lmax=lmax)
        self.nlee = nlee
        self.ls = self.cl_len.ls
        self.lmax = lmax
        self.lmin = 1
        self.freq_GHz = freq_GHz


        # Hyperparams for analytic QE calculation
        self.gauss_order = gauss_order
        self.nodes, self.weights = self.get_quad_nodes_weights(gauss_order, self.lmin, self.lmax)
        self.lnodes_grid, self.lpnodes_grid = np.meshgrid(self.nodes, self.nodes)

        self.massCut = massCut_Mvir #Convert from M_vir (which is what Alex uses) to M_200 (which is what the
                                    # Tinker mass function in hmvec uses) using the relation from White 01.

        self.nlev_t = nlev_t
        self.nlev_p = np.sqrt(2) * nlev_t
        self.beam_size = beam_size

        # Set up grid for Quicklens calculations
        self.nx = nx
        self.dx = dx_arcmin/60./180.*np.pi # pixel width in radians.
        self.nx_secbispec = nx_secbispec
        self.dx_secbispec = dx_arcmin_secbispec/60./180.*np.pi # pixel width in radians.
        self.pix = ql.maps.cfft(self.nx, self.dx)
        self.ivf_lib = None
        self.qest_lib = None

        self.tsz_filter = None

        self.MV_ILC_bool = MV_ILC_bool
        self.deproject_tSZ = deproject_tSZ
        self.deproject_CIB = deproject_CIB
        if not bare_bones:
            #Initialise sky model
            self.sky = cmb_ilc.CMBILC(freq_GHz*1e9, beam_size, nlev_t, fg=fg, atm=atm_fg, lMaxT=self.lmax)
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

    def get_quad_nodes_weights(self, gauss_order, a, b):
        # Get the nodes and weights for Gaussian quadrature of chosen order
        nodes_on_minus1to1, weights = roots_legendre(gauss_order)
        # We must now convert the nodes to our actual integration domain
        nodes = (b - a) / 2. * nodes_on_minus1to1 + (a + b) / 2.
        return nodes, (b - a) / 2. * weights

    def W_phi(self, lmax_clkk):
        # TODO: might want to specify cosmo here and elsewhere for ql
        clpp = ql.spec.get_camb_scalcl(None, lmax=lmax_clkk).clpp
        nlpp = self.get_nlpp(lmin=30, lmax=lmax_clkk, bin_width=30)
        return clpp / (clpp + nlpp)

    def W_E(self, lmax_clee):
        ells = np.arange(lmax_clee+1)
        if self.nlee is not None:
            self.clee_tot = self.sky.cmb[0, 0].flensedEE(ells) + self.nlee
        else:
            self.get_total_EE_power(lmax_clee)
        return np.nan_to_num(self.sky.cmb[0, 0].flensedEE(ells) / self.clee_tot)

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

    def get_total_EE_power(self, lmax):
        """
        Get total EE power from CMB, noise and fgs.
        Note that if both self.deproject_tSZ=1 and self.deproject_CIB=1, both are deprojected
        """
        ells = np.arange(lmax+1)
        #TODO: Why can't we get ells below 10 in cltt_tot?
        if len(self.freq_GHz)==1:
            self.clee_tot = self.sky.cmb[0, 0].ftotalEE(ells)
        else:
            nL = 201
            L = np.logspace(np.log10(self.lmin), np.log10(lmax), nL)
            # ToDo: sample better in L
            # TODO: add dust/sync deprojection options-- for now only MV
            f = lambda l: self.sky.powerIlcEE(self.sky.weightsIlcCmbEE(l), l)

            #TODO: turn zeros into infinities to avoid issues when dividing by this
            self.clee_tot = np.interp(ells, L, np.array(list(map(f, L))))
        # Avoid infinities when dividing by inverse variance
        #self.clee_tot[np.where(np.isnan(self.clee_tot))] = np.inf
        self.clee_tot = np.nan_to_num(self.clee_tot)

    def __getstate__(self):
        # this method is called when you are
        # going to pickle the class, to know what to pickle
        state = self.__dict__.copy()

        # don't pickle the parameter fun. otherwise will raise
        # AttributeError: Can't pickle local object 'Process.__init__.<locals>.<lambda>'
        del state['sky']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_weights_mat_total(self, ells_out):
        ''' Get the matrices needed for Gaussian quadrature of QE integral '''
        self.weights_mat_total = np.array([self.weights_mat_at_L(L) for L in ells_out])

    def weights_mat_at_L(self, L):
        '''
        Calculate the matrix to be used in the QE integration, i.e.,
        H(L). This is derived from
        W(L, l, l') \equiv -\Delta(l,l',L) l l' \left(\frac{L^2 + l'^2 - l^2}{2Ll'} \right)\left[1 - \left( \frac{L^2 +l'^2 -l^2}{2Ll'}\right)\right]^{-\frac{1}{2}}

        by sampling at the quadrature nodes in l and l' and  multiplying rows and columns by the quadrature weights w_i.

        We cache the outputs of this function as they are used recurrently at every mass and redshift step

        Inputs:
            - L = int. The L at which we are evaluating the QE reconstruction
        Returns:
            - W(L, l_i, l_j)
        '''
        return self.weights * self.weights[:, np.newaxis] * self.ell_dependence(L, self.lnodes_grid, self.lpnodes_grid)

    def ell_dependence(self, L, l, lp):
        '''
        Sample the kernel
        W(L, l, l') \equiv -\Delta(l,l',L) l l' \left(\frac{L^2 + l'^2 - l^2}{2Ll'} \right)\left[1 - \left( \frac{L^2 +l'^2 -l^2}{2Ll'}\right)\right]^{-\frac{1}{2}}

        Inputs:
            - L = int. The L at which we are evaluating the QE reconstruction
        Returns:
            - W(L, l, lp)
        '''
        L = np.asarray(L, dtype=int)  # Ensure L is an integer
        condition = (L + l >= lp) & (L + lp >= l) & (l + lp >= L)
        singular_condition = (L + l == lp) | (L + lp == l) | (l + lp == L)

        result = np.zeros_like(l, dtype=float)  # Initialize result array with appropriate dtype

        valid_indices = np.where(condition & ~singular_condition)
        #TODO: I've removed minus sign to get expected -ve sign at low L in prim bispec. What's up?
        result[valid_indices] = 2 * lp[valid_indices] * l[valid_indices] * (
                    (L ** 2 + lp[valid_indices] ** 2 - l[valid_indices] ** 2) / (2 * L * lp[valid_indices])) * (1 - (
                    (L ** 2 + lp[valid_indices] ** 2 - l[valid_indices] ** 2) / (2 * L * lp[valid_indices])) ** 2) ** (
                                    -0.5)
        return result

    def get_qe_norm(self, key='ptt'):
        """
        Calculate the QE normalisation as the reciprocal of the N^{(0)} bias
        Inputs:
            * (optional) key = String. The quadratic estimator key. Default is 'ptt' for TT
        """
        self.qest_lib = ql.sims.qest.library(self.cl_unl, self.cl_len, self.ivf_lib)
        self.qe_norm = self.qest_lib.get_qr(key)

    def get_nlpp(self, lmin=30, lmax=3000, bin_width=30):
        # TODO: adapt  the lmax of these bins to the lmax_out of hm_object
        # TODO: this N0 is not smooth. Find a better way to calculate
        ells = np.arange(lmax+1)
        lbins = np.arange(lmin, lmax, bin_width)
        norm = self.qe_norm.get_ml(lbins)
        # Extrapolate in clkk, for which we expect flatness at low L
        nlkk = np.interp(ells, norm.ls, np.nan_to_num(norm.ls**4/norm.specs['cl']))
        return np.nan_to_num(nlkk/ells**4)

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

def get_TT_qe(fftlog_way, ell_out, profile_leg1, qe_norm, pix, lmax, cltt_tot=None, ls=None, cltt_len=None,
              qest_lib=None, ivf_lib=None, profile_leg2=None, N_l=2*4096, lmin=0.000135, alpha=-1.3499,
              norm_bin_width=40, key='ptt', use_gauss=True, weights_mat_total=None, nodes=None):
    """
    Helper function to get the TT QE reconstruction for spherically-symmetric profiles using FFTlog
    Inputs:
        * fftlog_way = Bool. If true, use fftlog reconstruction. Otherwise use quicklens.
        * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
        * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
        * qe_norm = if fftlog_way=True, an experiment.qe_norm() instance.
                    otherwise, a 1D array containg the normalization of the TT QE at ell_out
        * pix = ql.maps.cfft() object. Contains numerical hyperparameters nx and dx
        * lmax = int. Maximum multipole used in the reconstruction
        * (optional) cltt_tot = 1d numpy array. Total power in observed TT fields. Needed if fftlog_way=1
        * (optional) ls = 1d numpy array. Multipoles at which cltt_tot is defined. Needed if fftlog_way=1
        * (optional) cltt_len = 1d numpy array. Lensed TT power spectrum at ls. Needed if fftlog_way=1
        * (optional) qest_lib = experiment.qest_lib() instance for quicklens lensing rec. Needed if fftlog_way=0
        * (optional) ivf_lib = experiment.ivf_lib() instance for quicklens lensing rec. Needed if fftlog_way=0
        * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
        * (optional) N_l = Integer (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
                           Needed if fftlog_way=1
        * (optional) lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values
                            (e.g., lmin=1e-4) to avoid ringing. Needed if fftlog_way=1
        * (optional) alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
                             Needed if fftlog_way=1
        * (optional) norm_bin_width = int. Bin width to use when taking spectra of the semi-analytic QE
                                      normalisation. Needed if fftlog_way=1
    Returns:
        * If fftlog_way=True, a 1D array with the unnormalised reconstruction at the multipoles specified in ell_out
        * (optional) key = String. The quadratic estimator key for quicklens. Default is 'ptt' for TT
        * (optional) use_gauss = Bool. If True, use Gaussian quad. Otherwise pyccl FFTlog
        * (optional) weights_mat_total = np array with dimensions (len(ells_out), len(nodes), len(nodes)). Required if
                                        use_gauss=True
        * (optional) nodes = 1D np array. The Gaussian-quadrature-determined ells at which to evaluate integrals. Needed
                                        if use_gauss=True
    """
    if profile_leg2 is None:
        profile_leg2 = profile_leg1
    if fftlog_way:
        assert(cltt_tot is not None and ls is not None and cltt_len is not None)
        al_F_1, al_F_2 = get_filtered_profiles_fftlog(profile_leg1, cltt_tot, ls, cltt_len, profile_leg2)
        # Calculate unnormalised QE
        if use_gauss==True:
            assert(weights_mat_total is not None and nodes is not None)
            F_1_array = al_F_1(nodes)
            F_2_array = al_F_2(nodes)
            unnorm_TT_qe = QE_via_quad(F_1_array, F_2_array, weights_mat_total)
        else:
            unnorm_TT_qe = unnorm_TT_qe_fftlog(al_F_1, al_F_2, N_l, lmin, alpha, lmax)(ell_out)
        # Apply a convention correction to match Quicklens
        conv_corr = 1/(2*np.pi)
        return conv_corr * np.nan_to_num( unnorm_TT_qe / qe_norm )
    else:
        assert(ivf_lib is not None and qest_lib is not None)
        tft1 = ql.spec.cl2cfft(profile_leg1, pix)
        # Apply filters and do lensing reconstruction
        t_filter = ivf_lib.get_fl().get_cffts()[0]
        tft1.fft *=t_filter.fft
        if profile_leg2 is None:
            tft2 = tft1.copy()
        else:
            tft2 = ql.spec.cl2cfft(profile_leg2, pix)
            tft2.fft *= t_filter.fft
        unnormalized_phi = qest_lib.get_qft(key, tft1, 0*tft1.copy(), 0*tft1.copy(),
                                                 tft2, 0*tft1.copy(), 0*tft1.copy())
        # In QL, the unnormalised reconstruction (obtained via eval_flatsky()) comes with a factor of sqrt(skyarea)
        A_sky = (pix.dx*pix.nx)**2
        #Normalize the reconstruction
        return np.nan_to_num(unnormalized_phi.fft[:,:] / qe_norm.fft[:,:]) /np.sqrt(A_sky)

def get_brute_force_unnorm_TT_qe(ell_out, profile_leg1, cltt_tot, ls, cltt_len, lmax,
                                 profile_leg2=None, max_workers=None):
    """
    Slow but sure method to calculate the 1D TT QE reconstruction.
    Scales as O(N^3), but useful as a cross-check of get_unnorm_TT_qe(fftlog_way=True)
    Inputs:
        * ell_out = 1D numpy array with the multipoles at which the reconstruction is wanted.
        * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
        * cltt_tot = 1d numpy array. Total power in observed TT fields.
        * ls = 1d numpy array. Multipoles at which cltt_tot is defined
        * cltt_len = 1d numpy array. Lensed TT power spectrum at ls.
        * lmax = int. Maximum multipole used in the reconstruction
        * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
        * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
    """
    al_F_1, al_F_2 = get_filtered_profiles_fftlog(profile_leg1, cltt_tot, ls, cltt_len, profile_leg2=profile_leg2)
    output_unnormalised_phi = np.zeros(ell_out.shape)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        n = len(ell_out)
        outputs = executor.map(int_func, n * [lmax], n * [ell_out], np.arange(n), n * [al_F_1], n * [al_F_2])

    for idx, outs in enumerate(outputs):
        output_unnormalised_phi[idx] = outs

    return output_unnormalised_phi

def int_func(lmax, ell_out, n, al_F_1, al_F_2):
    """
    Helper function to parallelize get_brute_force_unnorm_TT_qe()
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
        return l * al_F_1(l) * quad(inner_integrand, 1, lmax, args=(L, l))[0]
    L = ell_out[n]
    return quad(outer_integrand, 1, lmax, args=L)[0]/(2*np.pi)

def QE_via_quad(F_1_array, F_2_array, weights_mat):
    '''
    Unnormalized TT quadratic estimator

    \hat{\phi}(\vL) = 2 \int \frac{dl\,dl'}{2\pi} F_1(l) F_2(l') W(L, l, l')

    where

     W(L, l, l') \equiv - \Delta(l,l',L) l l' \left(\frac{L^2 + l'^2 - l^2}{2Ll'} \right)\left[1 - \left( \frac{L^2 +l'^2 -l^2}{2Ll'}\right)\right]^{-\frac{1}{2}}

    and \Delta(l,l',L) is the triangle condition. The double integral is calculated using Gaussian quadratures
    implemented in the form of matrix multiplication to harness the speed of numpy's BLAS library:

    F_1(l_i) H(L) F_2(l_i)

    where l_i are the Gaussian quadrature nodes (and similarly for F_2), and H(L) is a matrix derived from W(L, l_i, l_j)
    after it has absorbed the quadrature weights [as documented in weights_mat_total()]. It appears that only a few dozen nodes
    are needed for reasonable accuracy, but if more were required, one could explore using GPUs.

    Furthermore, we evaluate F_1(l_i) H(L) F_2(l_i) at the required L's via matrix-multiplication

    Inputs:
        - F_1_array = 1D np array. The filtered inputs of the QE evaluated at the Gaussian quadrature nodes
        - F_2_array = 1D np array. Same as F_1_array, but for F_2
        - weights_mat = 3D np array (len(L), len(ell), len(ellprime)) featuring the L, ell and ellprime dependence
    Returns:
        - The unnormalized lensing reconstruction at L
    '''
    return np.dot(np.matmul(F_2_array, weights_mat), F_1_array)/(2*np.pi)

def unnorm_TT_qe_fftlog(al_F_1, al_F_2, N_l, lmin, alpha, lmax):
    """
    Compute the unnormalised TT QE reconstruction for spherically symmetric profiles using FFTlog.
    Inputs:
        * al_F_1 = Interpolatable object from which to get F_1 (e.g., in eq. (7.9) of Lewis & Challinor 06)
                   at every multipole.
        * al_F_2 = Interpolatable object from which to get F_2 (e.g., in eq. (7.9) of Lewis & Challinor 06)
                   at every multipole.
        * N_l = Int (preferrably power of 2). Number of logarithmically-spaced samples FFTlog will use.
        * lmin = Float. lmin of the reconstruction. Recommend choosing (unphysical) small values
                            (e.g., lmin=1e-4) to avoid ringing
        * alpha = Float. FFTlog bias exponent. alpha=-1.35 seems to work fine for most applications.
        * lmax = int. Maximum multipole used in the reconstruction
    Returns:
        * An interp1d object into which you can plug in an array of ells to get the QE at those ells.
    """
    ell = np.logspace(np.log10(lmin), np.log10(lmax), N_l)

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

def get_filtered_profiles_fftlog(profile_leg1, cltt_tot, ls, cltt_len, profile_leg2=None):
    """
    Filter the profiles in the way of, e.g., eq. (7.9) of Lewis & Challinor 06.
    Inputs:
        * profile_leg1 = 1D numpy array. Projected, spherically-symmetric emission profile. Truncated at lmax.
        * cltt_tot = 1d numpy array. Total power in observed TT fields.
        * ls = 1d numpy array. Multipoles at which cltt_tot is defined
        * cltt_len = 1d numpy array. Lensed TT power spectrum at ls.
        * (optional) profile_leg2 = 1D numpy array. As profile_leg1, but for the other QE leg.
    Returns:
        * Interpolatable objects from which to get F_1 and F_2 at every multipole.
    """

    def smooth_low_monopoles(array):
        new = array[2:]
        return np.interp(np.arange(len(array)), np.arange(len(array))[2:], new)

    if profile_leg2 is None:
        profile_leg2 = profile_leg1
    F_1_of_l = smooth_low_monopoles(np.nan_to_num(profile_leg1 / cltt_tot))
    F_2_of_l = smooth_low_monopoles(np.nan_to_num(cltt_len * profile_leg2/ cltt_tot))

    al_F_1 = interp1d(ls, F_1_of_l, bounds_error=False,  fill_value='extrapolate')
    al_F_2 = interp1d(ls, F_2_of_l, bounds_error=False,  fill_value='extrapolate')
    return al_F_1, al_F_2