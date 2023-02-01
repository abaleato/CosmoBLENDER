# Script to generate secondary bispectrum bias on a cluster

import sys
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/lensing_rec_biases_code/')
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/')

import numpy as np
import matplotlib.pyplot as plt
from lensing_rec_biases_code import qest
from lensing_rec_biases_code import biases
import time
import tools as tls
import quicklens as ql

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t0 = time.time() # Time the run
    which_bias = 'tsz' #'tsz' or 'cib' or 'mixed'
    lmax = 3500  # Maximum ell for the reconstruction
    nx = 512
    dx_arcmin = 1.0 #* 2

    freq_GHz = np.array([27.3, 41.7, 93., 143., 225.,278.])  # [Hz]#np.array([27.e9, 39.e9, 93.e9, 145.e9, 225.e9, 280.e9])   # [Hz] #np.array([27.3e9, 41.7e9, 93.e9, 143.e9, 225.e9, 278.e9])   # [Hz]
    beam_size = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])  # [arcmin]
    nlev_t = np.array([52., 27., 5.8, 6.3, 15., 37.])  # [muK*arcmin]

    MV_ILC_bool = True

    # Set CIB halo model
    cib_model = 'planck13'  # 'vierro'
    survey_name = "built-in"
    # Initialise an experiment object. Store the list of params so we can later initialise it again within multiple processes
    massCut_Mvir=5e15
    experiment = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir, nx, dx_arcmin, freq_GHz=freq_GHz, MV_ILC_bool=MV_ILC_bool)

    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}  # Note that for now there is still cosmology dpendence in the cls defined within the experiment class

    nZs = 30
    nMasses = 30
    m_min = 2e11
    #which_bias = 'mixed'#'cib' #'tsz'

    # Initialise a halo model object for the calculation, using mostly default parameters
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams,  m_min=m_min, nZs=nZs, nMasses=nMasses, cib_model=cib_model)

    # Add the HOD for the galaxy sample that we will cross-correlating with lensing
    hm_calc.hcos.add_hod(name=survey_name, mthresh=10 ** 10.5 + hm_calc.hcos.zs * 0.)

    if which_bias=='tsz':
        hm_calc.get_tsz_cross_biases(experiment, survey_name=survey_name)
    elif which_bias=='cib':
        hm_calc.get_cib_cross_biases(experiment, survey_name=survey_name)
    elif which_bias=='mixed':
        hm_calc.get_mixed_cross_biases(experiment, survey_name=survey_name)

    plt.figure(figsize=(5, 5))
    scaling = 1#experiment.biases['ells'] ** 4 / (2 * np.pi)

    # Split into negative and positive parts for plotting convenience
    cross_w_gals_1h_pos, cross_w_gals_1h_neg = tls.split_positive_negative(
            experiment.biases[which_bias]['cross_w_gals']['1h'])
    cross_w_gals_2h_pos, cross_w_gals_2h_neg = tls.split_positive_negative(
            experiment.biases[which_bias]['cross_w_gals']['2h'])

    # Plot the signal clkg
    pgm_1h = hm_calc.hcos.get_power_1halo("nfw", survey_name)
    pgm_2h = hm_calc.hcos.get_power_2halo("nfw", survey_name)
    Pgm = pgm_1h + pgm_2h
    ells = np.linspace(2, 3000, 300)
    Cls = hm_calc.hcos.C_kg(ells, hm_calc.hcos.zs, hm_calc.hcos.ks, Pgm, gzs=0.8, lzs=1100.)
    plt.plot(ells, 0.05 * Cls, 'k')
    #plt.plot(experiment.cl_unl.ls, 0.05 * experiment.cl_unl.ls ** 4 * experiment.cl_unl.clpp / (2 * np.pi), 'k')

    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_1h_pos, color='b')
    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_1h_neg, color='b', ls='--')

    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_2h_pos, color='r')
    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_2h_neg, color='r', ls='--')

    #plt.plot(experiment.biases['ells'], scaling * (cross_w_gals_1h_pos + cross_w_gals_2h_pos), color='g')
    #plt.plot(experiment.biases['ells'], scaling * (cross_w_gals_1h_neg + cross_w_gals_2h_neg), color='g', ls='--')

    plt.yscale('log')
    plt.xlim([2, 3000])
    #plt.ylim([1e-12,1e-7])
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.ylabel(r'$C_{L}^{\delta_{g}\kappa}$')
    plt.xlabel(r'$L$')
    plt.savefig('testing_crosswgals_{}bias_nZs{}_nMasses{}.png'.format(which_bias, nZs, nMasses))
    plt.show()
    plt.close()
    t1 = time.time()
    print('time taken:', t1-t0)



