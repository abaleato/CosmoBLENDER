# We test out theory predictions agains the Websky sims https://mocks.cita.utoronto.ca/index.php/WebSky_Extragalactic_CMB_Mocks
# Note: I have not yet translated the curved-sky functionality of quicklens to Python3. So we need to run this notebook in a Python2 environment running the Python2 version of quicklens
import numpy as np
import quicklens as ql
import healpy as hp
import os


def shorten_alm(input_alm, lmax_new):
    lmax_old = hp.Alm.getlmax(len(input_alm))
    new_size = hp.Alm.getsize(lmax_new)
    output_alm = np.zeros(new_size, dtype=np.complex)

    index_in_new = np.arange(len(output_alm))
    l, m = hp.Alm.getlm(lmax_new, i=index_in_new)
    output_alm[index_in_new] = input_alm[hp.Alm.getidx(lmax_old, l, m)]
    return output_alm

def convert_phi_to_kappa(phi_map, ls, lmax, nside_out):
    phi_alm = hp.map2alm(phi_map, lmax=lmax, pol=False)
    kappa_alm = hp.sphtfunc.almxfl(phi_alm, 0.5 * ls ** 2)
    kappa_map = hp.alm2map(kappa_alm, nside_out)
    return kappa_map

def get_qe_rec_map(T_input1, T_input2):
    ''' Wrapper function to get the map QE reconstruction'''
    estimated_qft_vlm = qest_lib.get_qft_full_sky(key, T_input1, e_alm_filtered, b_alm_filtered,
                                                  T_input2, e_alm_filtered,
                                                  b_alm_filtered)
    g_alm, c_alm = ql.shts.util.vlm2alm(estimated_qft_vlm)

    normed_rec = hp.sphtfunc.almxfl(g_alm, np.nan_to_num(1. / norm))
    return hp.sphtfunc.alm2map(normed_rec, nside, lmax=lmax, pol=False)

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

def calc_nlqq(qest, clXX, clXY, clYY, flX, flY):
    """ Calculate N^0 noise bias given power spectra of some fields, using quicklens """
    clqq_fullsky = qest.fill_clqq(np.zeros(lmax + 1, dtype=np.complex), clXX * flX * flX, clXY * flX * flY,
                                  clYY * flY * flY)
    resp_fullsky = qest.fill_resp(qest, np.zeros(lmax + 1, dtype=np.complex), flX, flY)
    nlqq_fullsky = clqq_fullsky / resp_fullsky ** 2
    return nlqq_fullsky

def get_fg_N0(measured_fg_ps, nltt):
    """ Get the contribution to N^0 from the fg 4-pt function.
        - Input:
            * measured_fg_ps = np array. Foreground power spectrum measured from the sims.
            * nltt = np array. Intrument noise power spectrum.
    """
    # signal spectra
    sltt = cl_len.cltt
    # signal+noise spectra
    cltt = sltt + nltt
    # filter functions
    flt = np.zeros(lmax + 1);
    flt[2:] = 1. / cltt[2:]
    # intialize quadratic estimators
    qest_TT = ql.qest.lens.phi_TT(sltt)

    nlpp_TT_fullsky = calc_nlqq(qest_TT, measured_fg_ps, measured_fg_ps, measured_fg_ps, flt, flt)
    return nlpp_TT_fullsky

def load_tsz(nside, lmax, freq_GHz):
    """ Load Websky tSZ sim. Return it in uK units"""
    global websky_dir
    T_CMB = 2.73
    tsz_scaling = scale_sz(freq_GHz) * T_CMB * 1e6
    tsz_map = tsz_scaling * hp.remove_monopole(
        hp.ud_grade(hp.read_map(websky_dir + 'tsz_2048.fits'), nside_out=nside))
    tsz_alm = hp.map2alm(tsz_map, lmax=lmax)
    return tsz_map, tsz_alm

def load_cib(nside, lmax, freq_CIB_GHz, masking_threshold_mJy=None):
    """ Load Websky CIB sim. Return it in uK units"""
    global websky_dir

    cib_map_MJy_sr = hp.read_map(websky_dir + 'cib_nu0143.fits')

    # some useful constants
    h_over_k = 0.047992447 #:[K*s]
    h_over_c2 = 7.3724972E-4 #: [MJy/sr/GHz^3]
    k3_over_c2h2 = 6.66954097 # :[MJy/sr]
    c2_over_k = 65.096595 # :[K*GHz^2/MJy/sr

    def diff_black_body(nu, T):
        """
        differential BB spectrum [MJy/sr] with a temperature T [K] at frequency nu [GHz]
        ***multiply by dT/T to get dI_nu***

        :param nu: observed frequency [GHz]
        :param T: thermodynamic temperature [K]
        :return: dI_nu*T/dT [MJy/sr]
        """
        
        x = h_over_k * nu / T  # nu: [GHz] ; T: [K]
        f = x ** 4 * np.exp(x) / np.expm1(x) ** 2

        return 2 * k3_over_c2h2 * T ** 3 * f

    T_cmb = 2.7255 # [K]

    from_MJypersr_to_K = T_cmb/diff_black_body(freq_CIB_GHz, T_cmb)
    from_mJypersr_to_muK = from_MJypersr_to_K * 1e6 / 1e9
    cib_map = cib_map_MJy_sr * from_MJypersr_to_K * 1e6 # [uK]

    if masking_threshold_mJy is not None:
        # Now convert the flux cut to uK
        pix_area = (4*np.pi)/len(cib_map)
        flux_cut_mJy_per_sr = masking_threshold_mJy / pix_area
        flux_cut_muK = flux_cut_mJy_per_sr * from_mJypersr_to_muK

        cib_map_masked = cib_map.copy()

        masked_idx = np.logical_or((cib_map < -flux_cut_muK), (cib_map > flux_cut_muK))
        cib_map_masked[masked_idx] = np.mean(cib_map[~masked_idx])
        print('masked {}% of pixels'.format(100*np.sum(masked_idx)/len(cib_map)))

        cib_map = cib_map_masked

    cib_alm = hp.map2alm(hp.ud_grade(cib_map, nside_out=nside), lmax=lmax)

    return cib_map, cib_alm

if __name__ == '__main__':
    which_bias = 'CIB' # 'tSZ' or 'CIB' or 'all'
    lmax = 3000
    nside = 2048

    freq_tsz = 150. # in GHz
    freq_CIB = 150. # in GHz
    # Masking threshold for CIB point sources
    masking_threshold_mJy = 1.2 # in mJy

    key = 'ptt'  # 'peb' # 'ptt'
    # Experiment setup
    nlev_t = 15.0  # 17.0 #temperature noise level, in uK.arcmin.
    beam_size = 1.4  # 1. #arcmin
    npad = 2

    # Load the Websky sims
    websky_dir = '/pscratch/sd/a/ab2368/websky/'
    # Create new directory to save files
    output_dir = websky_dir+'biases_from_websky_lmax{}_nside{}_nlevt{}_beamarcmin{}/'.format(lmax, nside, nlev_t, beam_size)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    masking_suffix = ''
    if which_bias == 'tSZ':
        fg_map, fg_alm = load_tsz(nside, lmax, freq_tsz)
    elif which_bias == 'CIB':
        fg_map, fg_alm = load_cib(nside, lmax, freq_CIB, masking_threshold_mJy)
        if masking_threshold_mJy is not None:
            masking_suffix = '_{}cut'.format(masking_threshold_mJy)
    elif which_bias == 'all':
        tsz_map, tsz_alm = load_tsz(nside, lmax, freq_tsz)
        cib_map, cib_alm = load_cib(nside, lmax, freq_CIB, masking_threshold_mJy)
        # Add the maps together
        fg_map = cib_map + tsz_map
        fg_alm = cib_alm + tsz_alm

    # CMB
    websky_cmb_alm = shorten_alm(hp.read_alm(websky_dir + 'lensed_alm.fits'), lmax)
    websky_unlensed_cmb_alm = shorten_alm(hp.read_alm(websky_dir + 'unlensed_alm.fits'), lmax)

    # Kappa
    true_kappa = hp.read_map(websky_dir + 'kap.fits')
    # First order lensing correction to the temperature. Calculated in get_1storder_lensedT.py
    if os.path.exists(websky_dir+'g_T_alm_first_order_in_lensing_lmax{}.npy'.format(lmax)):
        g_T_alm_1storder = np.load(websky_dir+'g_T_alm_first_order_in_lensing_lmax{}.npy'.format(lmax))
    else:
        g_T_alm_1storder = np.load(websky_dir+'g_T_alm_first_order_in_lensing_lmax{}.npy'.format(3000))

    # Set E and B to zeros bc we're focusing on TT reconstruction
    Ealm = Balm = np.zeros(websky_cmb_alm.shape)

    # Initialise quadratic estimators
    cl_unl = ql.spec.get_camb_scalcl(lmax=lmax)
    cl_len = ql.spec.get_camb_lensedcl(lmax=lmax)

    bl = ql.spec.bl(beam_size, lmax)  # beam transfer function.

    nlev_p = np.sqrt(2) * nlev_t  # polarization noise level, in uK.arcmin.
    nltt = (np.pi / 180. / 60. * nlev_t) ** 2 / bl ** 2
    nlee = nlbb = (np.pi / 180. / 60. * nlev_p) ** 2 / bl ** 2

    beam = ql.spec.clmat_teb(ql.util.dictobj({'lmax': lmax, 'cltt': bl, 'clee': bl, 'clbb': bl}))
    cl_theory = ql.spec.clmat_teb(
        ql.util.dictobj({'lmax': lmax, 'cltt': cl_len.cltt, 'clee': cl_len.clee, 'clbb': cl_len.clbb}))

    ivf_lib = ql.sims.ivf.library_diag_full_sky(cl_theory, beam, nlev_t=nlev_t, nlev_p=nlev_p)

    qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, npad=npad)
    norm = qest_lib.get_qr_full_sky(key)

    # Wiener filter the inputs of the QE
    cmbT_alm_filtered = ivf_lib.ivf_alm_array(websky_cmb_alm, 'cltt')
    cmbT_unlensed_alm_filtered = ivf_lib.ivf_alm_array(websky_unlensed_cmb_alm, 'cltt')
    cmbT_1storder_alm_filtered = ivf_lib.ivf_alm_array(g_T_alm_1storder, 'cltt')
    fgT_alm_filtered = ivf_lib.ivf_alm_array(fg_alm, 'cltt')
    e_alm_filtered = ivf_lib.ivf_alm_array(Ealm, 'clee')
    b_alm_filtered = ivf_lib.ivf_alm_array(Balm, 'clbb')

    ells = np.arange(lmax + 1)
    if not os.path.exists(output_dir+'cmbonly_rec_map.npy'):
        # Perform the reconstruction with CMB ONLY AS INPUT
        cmbonly_rec_map = get_qe_rec_map(cmbT_alm_filtered, cmbT_alm_filtered)
        np.save(output_dir + 'cmbonly_rec_map', cmbonly_rec_map)
    else:
        cmbonly_rec_map = np.load(output_dir+'cmbonly_rec_map.npy')

    if not os.path.exists(output_dir+'{}only_rec_map{}.npy'.format(which_bias, masking_suffix)):
        # Perform the reconstruction with FOREGROUND ONLY AS INPUT
        fgonly_rec_map = get_qe_rec_map(fgT_alm_filtered, fgT_alm_filtered)
        np.save(output_dir + '{}only_rec_map'.format(which_bias), fgonly_rec_map)
    else:
        fgonly_rec_map = np.load(output_dir+'{}only_rec_map{}.npy'.format(which_bias, masking_suffix))

    # Perform the reconstruction with MIXED FOREGROUND AND CMB INPUTS
    # Zeroth order T and fg
    if not os.path.exists(output_dir+'mixed_unlensedcmb{}_rec_map{}.npy'.format(which_bias, masking_suffix)):
        mixed_unlensedandfg_rec_map = get_qe_rec_map(cmbT_unlensed_alm_filtered, fgT_alm_filtered)
        np.save(output_dir + 'mixed_unlensedcmb{}_rec_map{}'.format(which_bias, masking_suffix), mixed_unlensedandfg_rec_map)
    else:
        mixed_unlensedandfg_rec_map = np.load(output_dir+'mixed_unlensedcmb{}_rec_map{}.npy'.format(which_bias, masking_suffix))

    if not os.path.exists(output_dir+'mixed_1stordercmb{}_rec_map{}.npy'.format(which_bias, masking_suffix)):
        # First order T and fg
        mixed_1storderandfg_rec_map = get_qe_rec_map(cmbT_1storder_alm_filtered, fgT_alm_filtered)
        np.save(output_dir + 'mixed_1stordercmb{}_rec_map{}'.format(which_bias, masking_suffix), mixed_1storderandfg_rec_map)
    else:
        mixed_1storderandfg_rec_map = np.load(output_dir+'mixed_1stordercmb{}_rec_map{}.npy'.format(which_bias, masking_suffix))

    # Take spectra

    # Get kappa from phi
    cmbonly_rec_kappa_map = convert_phi_to_kappa(cmbonly_rec_map, ells, lmax, nside)
    fgonly_rec_kappa_map = convert_phi_to_kappa(fgonly_rec_map, ells, lmax, nside)
    mixed_unlensedandfg_rec_kappa_map = convert_phi_to_kappa(mixed_unlensedandfg_rec_map, ells, lmax, nside)
    mixed_1storderandfg_rec_kappa_map = convert_phi_to_kappa(mixed_1storderandfg_rec_map, ells, lmax, nside)

    if not os.path.exists(output_dir+'cmbrec_x_cmbrec.npy'):
        cmbrec_x_cmbrec = hp.sphtfunc.anafast(cmbonly_rec_kappa_map, lmax=lmax, pol=False)
        np.save(output_dir + 'cmbrec_x_cmbrec', cmbrec_x_cmbrec)

    if not os.path.exists(output_dir+'true_x_cmbrec.npy'):
        true_x_cmbrec = hp.sphtfunc.anafast(true_kappa, cmbonly_rec_kappa_map, lmax=lmax, pol=False)
        np.save(output_dir + 'true_x_cmbrec', true_x_cmbrec)

    if not os.path.exists(output_dir+'true_x_true.npy'):
        true_x_true = hp.sphtfunc.anafast(true_kappa, true_kappa, lmax=lmax, pol=False)
        np.save(output_dir + 'true_x_true', true_x_true)

    if not os.path.exists(output_dir + '{}onlyrec_x_{}onlyrec{}.npy'.format(which_bias, which_bias, masking_suffix)):
        # Trispectrum bias
        fgonlyrec_x_fgonlyrec = hp.sphtfunc.anafast(fgonly_rec_kappa_map, lmax=lmax, pol=False)
        np.save(output_dir + '{}onlyrec_x_{}onlyrec{}'.format(which_bias, which_bias, masking_suffix), fgonlyrec_x_fgonlyrec)

    if not os.path.exists(output_dir + 'N0_TT_fullsky_{}{}.npy'.format(which_bias, masking_suffix)):
        # Measure the power spectrum of the foregrounds (to then calculate N^0 contribution and subtract from trispectrum estimate
        measured_fg_ps = hp.sphtfunc.anafast(fg_map, lmax=lmax, pol=False)
        pixwin = hp.pixwin(nside)[:lmax+1]
        nlpp_TT_fullsky = get_fg_N0(measured_fg_ps/pixwin**2, nltt)
        np.save(output_dir + 'N0_TT_fullsky_{}{}'.format(which_bias, masking_suffix), nlpp_TT_fullsky)

    if not os.path.exists(output_dir + 'cmbonly_x_{}onlyrec{}.npy'.format(which_bias, masking_suffix)):
        # Primary bispectrum bias. We replace cmbonly_rec_kappa_map with true_kappa to avoid noise
        cmbonly_x_fgonlyrec = hp.sphtfunc.anafast(true_kappa, fgonly_rec_kappa_map, lmax=lmax, pol=False)
        np.save(output_dir + 'cmbonly_x_{}onlyrec{}'.format(which_bias, masking_suffix), cmbonly_x_fgonlyrec)

    if not os.path.exists(output_dir + 'mixed_cmb{}_rec_x_mixed_cmb{}_rec{}.npy'.format(which_bias, which_bias, masking_suffix)):
        mixedrec_x_mixedrec = hp.sphtfunc.anafast(mixed_unlensedandfg_rec_kappa_map,
                                              mixed_1storderandfg_rec_kappa_map, lmax=lmax, pol=False)
        np.save(output_dir +  'mixed_cmb{}_rec_x_mixed_cmb{}_rec{}'.format(which_bias, which_bias, masking_suffix), mixedrec_x_mixedrec)
