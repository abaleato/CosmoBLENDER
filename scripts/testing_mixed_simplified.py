import sys
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/lensing_rec_biases_code/')
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/')
import numpy as np
import matplotlib.pyplot as plt
from lensing_rec_biases_code import qest
from lensing_rec_biases_code import biases
import time
import tools as tls

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t0 = time.time() # Time the run
    nlev_t = 18.  # uK arcmin
    beam_size = 1.  # arcmin
    lmax = 3000  # Maximum ell for the reconstruction
    nx = 512
    dx_arcmin = 1.0 #* 2

    # Set CIB halo model
    cib_model = 'planck13'  # 'vierro'
    # Initialise an experiment object. Store the list of params so we can later initialise it again within multiple processes
    massCut_Mvir=5e15
    exp_param_list = [nlev_t, beam_size, lmax, massCut_Mvir, nx, dx_arcmin]
    SPT_5e15 = qest.experiment(*exp_param_list)
    experiment = SPT_5e15

    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}  # Note that for now there is still cosmology dpendence in the cls defined within the experiment class

    nZs = 10
    nMasses = 10
    #bin_width_out_second_bispec_bias = 1000
    #which_bias = 'mixed'#'cib' #'tsz'

    # Initialise a halo model object for the calculation, using mostly default parameters
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, cib_model=cib_model)

    plt.figure(figsize=(5, 5))

    fftlog_way = True
    ells_out = np.arange(hm_calc.lmax_out + 1)
    scaling = np.arange(hm_calc.lmax_out + 1) ** 4 / (2 * np.pi)

    nx = hm_calc.lmax_out + 1 if fftlog_way else experiment.pix.nx

    oneHalo_cross_tsz = np.zeros([nx, hm_calc.nZs]) + 0j if fftlog_way else np.zeros([nx, nx, hm_calc.nZs]) + 0j
    oneHalo_cross_cib = oneHalo_cross_tsz.copy();
    oneHalo_cross_mixed = oneHalo_cross_tsz.copy()

    hod_fact_1gal = hm_calc.get_hod_factorial(1, experiment)
    hod_fact_2gal = hm_calc.get_hod_factorial(2, experiment)

    for i, z in enumerate(hm_calc.hcos.zs):
        integrand_oneHalo_cross_tsz = np.zeros([nx, hm_calc.nMasses]) + 0j if fftlog_way else np.zeros(
            [nx, nx, hm_calc.nMasses]) + 0j
        integrand_oneHalo_cross_cib = integrand_oneHalo_cross_tsz.copy();
        integrand_oneHalo_cross_mixed = integrand_oneHalo_cross_tsz.copy();

        for j, m in enumerate(hm_calc.hcos.ms):
            if m > experiment.massCut: continue
            u = tls.pkToPell(hm_calc.hcos.comoving_radial_distance(hm_calc.hcos.zs[i]), hm_calc.hcos.ks,
                             hm_calc.hcos.uk_profiles['nfw'][i, j] \
                             * (1 - np.exp(-(hm_calc.hcos.ks / hm_calc.hcos.p['kstar_damping']))),
                             ellmax=experiment.lmax)
            y = tls.pkToPell(hm_calc.hcos.comoving_radial_distance(hm_calc.hcos.zs[i]), hm_calc.hcos.ks,
                             hm_calc.hcos.pk_profiles['y'][i, j] \
                             * (1 - np.exp(-(hm_calc.hcos.ks / hm_calc.hcos.p['kstar_damping']))),
                             ellmax=experiment.lmax)
            # Get the kappa map
            kap = tls.pkToPell(hm_calc.hcos.comoving_radial_distance(hm_calc.hcos.zs[i]), hm_calc.hcos.ks,
                               hm_calc.hcos.uk_profiles['nfw'][i, j] \
                               * hm_calc.hcos.lensing_window(hm_calc.hcos.zs[i], 1100.), ellmax=hm_calc.lmax_out)

            kfft = kap * hm_calc.ms_rescaled[j] / (1 + hm_calc.hcos.zs[i]) ** 3 if fftlog_way else ql.spec.cl2cfft(kap,
                                                                                                                   experiment.pix).fft * \
                                                                                                   hm_calc.ms_rescaled[
                                                                                                       j] / (1 +
                                                                                                             hm_calc.hcos.zs[
                                                                                                                 i]) ** 3

            phi_estimate_cfft_yy = experiment.get_TT_qe(fftlog_way, ells_out, y, y)
            phi_estimate_cfft_uy = experiment.get_TT_qe(fftlog_way, ells_out, u, y)
            phi_estimate_cfft_uu = experiment.get_TT_qe(fftlog_way, ells_out, u, u)

            # Accumulate the integrands
            integrand_oneHalo_cross_tsz[..., j] = hm_calc.hcos.nzm[i, j] * phi_estimate_cfft_yy * np.conjugate(kfft)
            integrand_oneHalo_cross_mixed[..., j] = hm_calc.hcos.nzm[i, j] * phi_estimate_cfft_uy * np.conjugate(kfft) * \
                                                    hod_fact_1gal[i, j]
            integrand_oneHalo_cross_cib[..., j] = hm_calc.hcos.nzm[i, j] * phi_estimate_cfft_uu * np.conjugate(kfft) * \
                                                  hod_fact_2gal[i, j]

        oneHalo_cross_tsz[..., i] = np.trapz(integrand_oneHalo_cross_tsz, hm_calc.hcos.ms, axis=-1)
        oneHalo_cross_mixed[..., i] = np.trapz(integrand_oneHalo_cross_mixed, hm_calc.hcos.ms, axis=-1)
        oneHalo_cross_cib[..., i] = np.trapz(integrand_oneHalo_cross_cib, hm_calc.hcos.ms, axis=-1)

    conversion_factor = np.nan_to_num(1 / (0.5 * ells_out * (ells_out + 1)))

    kyy_integrand = 2 * hm_calc.hcos.comoving_radial_distance(hm_calc.hcos.zs) ** -4 * hm_calc.hcos.h_of_z(
        hm_calc.hcos.zs) ** 2
    kIy_integrand = 2 * hm_calc.hcos.comoving_radial_distance(hm_calc.hcos.zs) ** -4 * hm_calc.hcos.h_of_z(
        hm_calc.hcos.zs) * (1 + hm_calc.hcos.zs) ** -1 * 2
    kII_integrand = 2 * hm_calc.hcos.comoving_radial_distance(hm_calc.hcos.zs) ** -4 * (1 + hm_calc.hcos.zs) ** -2

    tsz = conversion_factor * np.trapz(kyy_integrand * oneHalo_cross_tsz, hm_calc.hcos.zs, axis=-1) * tls.scale_sz(
        experiment.freq_GHz) ** 2 * hm_calc.T_CMB ** 2
    mixed = conversion_factor * np.trapz(kIy_integrand * oneHalo_cross_mixed, hm_calc.hcos.zs, axis=-1) * tls.scale_sz(
        experiment.freq_GHz) * hm_calc.T_CMB
    cib = conversion_factor * np.trapz(kII_integrand * oneHalo_cross_cib, hm_calc.hcos.zs, axis=-1)

    plt.plot(scaling * tsz, 'b')
    plt.plot(-scaling * tsz, 'b', ls='--')

    plt.plot(scaling * cib, 'r')
    plt.plot(-scaling * cib, 'r', ls='--')

    plt.plot(scaling * mixed, 'gold')
    plt.plot(-scaling * mixed, 'gold', ls='--')

    #plt.plot(scaling * (tsz + cib + mixed), 'green')
    #plt.plot(-scaling * (tsz + cib + mixed), 'green', ls='--')

    plt.yscale('log')
    plt.xlim([2, 3000])
    plt.ylim([1e-12, 1e-7])
    plt.show()