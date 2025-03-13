# Script to generate the first order correction to the temperature due to lensing
import numpy as np
import healpy as hp
import csbt


def shorten_alm(input_alm, lmax_new):
    lmax_old = hp.Alm.getlmax(len(input_alm))
    new_size = hp.Alm.getsize(lmax_new)
    output_alm = np.zeros(new_size, dtype=np.complex)

    index_in_new = np.arange(len(output_alm))
    l, m = hp.Alm.getlm(lmax_new, i=index_in_new)
    output_alm[index_in_new] = input_alm[hp.Alm.getidx(lmax_old, l, m)]
    return output_alm

def kappa_map_phi_to_phi_alm(kappa_map, ls, lmax):
    kappa_alm = hp.map2alm(kappa_map, lmax=lmax, pol=False)
    conversion_factor = 2. / ls ** 2
    conversion_factor[0:2] = 0
    phi_alm = hp.sphtfunc.almxfl(kappa_alm, conversion_factor)
    return phi_alm


if __name__ == '__main__':
    lmax = 3000 #5000
    which_sim = 'agora' # 'websky' or 'agora'

    if which_sim == 'websky':
        out_dir = '/pscratch/sd/a/ab2368/websky/'
        sim_dir = out_dir
        kappa_dir = sim_dir + 'kap.fits'
        unl_alm_path = sim_dir + 'unlensed_alm.fits'
    elif which_sim == 'agora':
        out_dir = '/pscratch/sd/a/ab2368/agora/'
        sim_dir = '/global/cfs/projectdirs/cmb/data/agora/components/'
        kappa_dir = sim_dir+'cmbkappa/agora_born_cmbkappa_highzadded_lowLcorrected.fits'
        unl_alm_path = sim_dir+'cmb/unl/teb1/agora_phiG_teb1_seed1.alm'
        

    unlensed_cmb_alm = shorten_alm(hp.read_alm(unl_alm_path, hdu=[1]), lmax)
    true_kappa = hp.read_map(kappa_dir)
    phi_alm = kappa_map_phi_to_phi_alm(true_kappa, np.arange(lmax), lmax)

    T_template = csbt.weights.T_template_weights(np.ones(lmax + 1)).eval_fullsky(phi_alm, unlensed_cmb_alm)
    g_T_alm, c_T_alm = csbt.shts.util.vlm2alm(T_template)
    np.save(out_dir+'g_T_alm_first_order_in_lensing_lmax{}'.format(lmax), g_T_alm)
    print('Done here')

