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
        hp.ud_grade(hp.read_map(websky_dir + 'tsz.fits'), nside_out=nside))
    tsz_alm = hp.map2alm(tsz_map, lmax=lmax)
    return tsz_map, tsz_alm

def load_cib(nside, lmax, freq_CIB, masking_threshold_mJy):
    """ Load Websky CIB sim. Return it in uK units"""
    global websky_dir
    if freq_CIB==143.:
        mJy_per_sr_to_uK_at143GHz = 1e-3 / (379.93197)  # For infinitely-narrow bands
    elif freq_CIB==145.:
        mJy_per_sr_to_uK_at145GHz = 1e-3 / (385.39618) # For infinitely-narrow bands
    else:
        print 'Please hard-code unit conversion for CIB at this frequency'

    cib_map_MJy = hp.pixelfunc.remove_monopole(
        hp.ud_grade(hp.read_map(websky_dir + 'cib_nu0143.fits'), nside_out=nside))

    # Mask point sources above masking_threshold_mJy
    cib_map_mJy = 1e9 * cib_map_MJy
    # Flag the entries to be masked
    cib_map_mJy[cib_map_mJy > masking_threshold_mJy] = 0  # hp.UNSEEN
    # Mask using healpy functionality
    masked_cib_map_mJy = cib_map_mJy  # hp.pixelfunc.ma(cib_map_mJy)
    # Convert from mJy/sr to uK
    cib_map = mJy_per_sr_to_uK_at143GHz * masked_cib_map_mJy
    cib_alm = hp.map2alm(cib_map, lmax=lmax)
    return cib_map, cib_alm

if __name__ == '__main__':
    which_bias = 'all' # 'tSZ' or 'CIB' or 'all'
    lmax = 3000
    nside = 1024

    freq_tsz = 143. # in GHz
    freq_CIB = 143. # in GHz
    # Masking threshold for CIB point sources
    masking_threshold_mJy = 5 # in mJy

    key = 'ptt'  # 'peb' # 'ptt'
    # Experiment setup
    nlev_t = 18.0  # 17.0 #temperature noise level, in uK.arcmin.
    beam_size = 1.  # 1. #arcmin
    npad = 2

    scripts_dir = '/Users/antonbaleatolizancos/Projects/lensing_rec_biases/scripts/'
    # Load the Websky sims
    websky_dir = '/Volumes/TOSHIBA_EXT/data/sims/websky/'
    # Create new directory to save files
    output_dir = 'biases_from_websky_lmax{}_nside{}_nlevt{}_beamarcmin{}/'.format(lmax, nside, nlev_t, beam_size)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if which_bias == 'tSZ':
        fg_map, fg_alm = load_tsz(nside, lmax, freq_tsz)
    elif which_bias == 'CIB':
        fg_map, fg_alm = load_cib(nside, lmax, freq_CIB, masking_threshold_mJy)
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
    if os.path.exists(scripts_dir+'g_T_alm_first_order_in_lensing_lmax{}.npy'.format(lmax)):
        g_T_alm_1storder = np.load(scripts_dir+'g_T_alm_first_order_in_lensing_lmax{}.npy'.format(lmax))
    else:
        g_T_alm_1storder = np.load(scripts_dir+'g_T_alm_first_order_in_lensing_lmax{}.npy'.format(3000))

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

    # Measure the power spectrum of the foregrounds (to then calculate N^0 contribution and subtract from trispectrum estimate
    measured_fg_ps = hp.sphtfunc.anafast(fg_map, lmax=lmax, pol=False) # Notice we don't include tsz_scaling.
                                                                           # We correct at the end, when plotting
    nlpp_TT_fullsky = get_fg_N0(measured_fg_ps, nltt)
    np.save(output_dir + 'N0_TT_fullsky_{}'.format(which_bias), nlpp_TT_fullsky)

    # Wiener filter the inputs of the QE
    cmbT_alm_filtered = ivf_lib.ivf_alm_array(websky_cmb_alm, 'cltt')
    cmbT_unlensed_alm_filtered = ivf_lib.ivf_alm_array(websky_unlensed_cmb_alm, 'cltt')
    cmbT_1storder_alm_filtered = ivf_lib.ivf_alm_array(g_T_alm_1storder, 'cltt')
    fgT_alm_filtered = ivf_lib.ivf_alm_array(fg_alm, 'cltt')
    e_alm_filtered = ivf_lib.ivf_alm_array(Ealm, 'clee')
    b_alm_filtered = ivf_lib.ivf_alm_array(Balm, 'clbb')

    ells = np.arange(lmax + 1)
    # Perform the reconstruction with CMB ONLY AS INPUT
    cmbonly_rec_map = get_qe_rec_map(cmbT_alm_filtered, cmbT_alm_filtered)
    np.save(output_dir + 'cmbonly_rec_map', cmbonly_rec_map)

    # Perform the reconstruction with FOREGROUND ONLY AS INPUT
    fgonly_rec_map = get_qe_rec_map(fgT_alm_filtered, fgT_alm_filtered)
    np.save(output_dir + '{}only_rec_map'.format(which_bias), fgonly_rec_map)

    # Perform the reconstruction with MIXED FOREGROUND AND CMB INPUTS
    # Zeroth order T and fg
    mixed_unlensedandfg_rec_map = get_qe_rec_map(cmbT_unlensed_alm_filtered, fgT_alm_filtered)
    np.save(output_dir + 'mixed_unlensedcmb{}_rec_map'.format(which_bias), mixed_unlensedandfg_rec_map)
    # First order T and fg
    mixed_1storderandfg_rec_map = get_qe_rec_map(cmbT_1storder_alm_filtered, fgT_alm_filtered)
    np.save(output_dir + 'mixed_1stordercmb{}_rec_map'.format(which_bias), mixed_1storderandfg_rec_map)

    # Take spectra

    # Get kappa from phi
    cmbonly_rec_kappa_map = convert_phi_to_kappa(cmbonly_rec_map, ells, lmax, nside)
    fgonly_rec_kappa_map = convert_phi_to_kappa(fgonly_rec_map, ells, lmax, nside)
    mixed_unlensedandfg_rec_kappa_map = convert_phi_to_kappa(mixed_unlensedandfg_rec_map, ells, lmax, nside)
    mixed_1storderandfg_rec_kappa_map = convert_phi_to_kappa(mixed_1storderandfg_rec_map, ells, lmax, nside)

    # CMB lensing reconstruction auto
    cmbrec_x_cmbrec = hp.sphtfunc.anafast(cmbonly_rec_kappa_map, lmax=lmax, pol=False)
    # CMB lensing reconstruction cross true kappa
    true_x_cmbrec = hp.sphtfunc.anafast(true_kappa, cmbonly_rec_kappa_map, lmax=lmax, pol=False)
    # True kappa auto
    true_x_true = hp.sphtfunc.anafast(true_kappa, true_kappa, lmax=lmax, pol=False)

    # Trispectrum bias
    fgonlyrec_x_fgonlyrec = hp.sphtfunc.anafast(fgonly_rec_kappa_map, lmax=lmax, pol=False)
    # Primary bispectrum bias. We replace cmbonly_rec_kappa_map with true_kappa to avoid noise
    cmbonly_x_fgonlyrec = hp.sphtfunc.anafast(true_kappa, fgonly_rec_kappa_map, lmax=lmax, pol=False)
    # Secondary bispectrum bias
    mixedrec_x_mixedrec = hp.sphtfunc.anafast(mixed_unlensedandfg_rec_kappa_map,
                                              mixed_1storderandfg_rec_kappa_map, lmax=lmax, pol=False)

    # SAVE SPECTRA
    np.save(output_dir + 'cmbrec_x_cmbrec', cmbrec_x_cmbrec)
    np.save(output_dir + 'true_x_cmbrec', true_x_cmbrec)
    np.save(output_dir + 'true_x_true', true_x_true)
    np.save(output_dir + '{}onlyrec_x_{}onlyrec'.format(which_bias, which_bias), fgonlyrec_x_fgonlyrec)
    np.save(output_dir + 'cmbonly_x_{}onlyrec'.format(which_bias), cmbonly_x_fgonlyrec)
    np.save(output_dir + 'mixed_cmb{}_rec_x_mixed_cmb{}_rec'.format(which_bias, which_bias), mixedrec_x_mixedrec)