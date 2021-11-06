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
    which_bias = 'cib' #'tsz' or 'cib'
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

    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}  # Note that for now there is still cosmology dpendence in the cls defined within the experiment class

    nZs = 10
    nMasses = 10
    #bin_width_out_second_bispec_bias = 1000
    #which_bias = 'mixed'#'cib' #'tsz'

    # Initialise a halo model object for the calculation, using mostly default parameters
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, cib_model=cib_model)

    experiment = SPT_5e15

    if which_bias=='tsz':
        hm_calc.get_tsz_cross_biases(SPT_5e15)
    elif which_bias=='cib':
        hm_calc.get_cib_cross_biases(SPT_5e15)

    plt.figure(figsize=(5, 5))
    scaling = experiment.biases['ells'] ** 4 / (2 * np.pi)

    # Split into negative and positive parts for plotting convenience
    cross_w_gals_1h_pos, cross_w_gals_1h_neg = tls.split_positive_negative(
            experiment.biases[which_bias]['cross_w_gals']['1h'])
    cross_w_gals_2h_pos, cross_w_gals_2h_neg = tls.split_positive_negative(
            experiment.biases[which_bias]['cross_w_gals']['2h'])

    plt.plot(experiment.cl_unl.ls, 0.05 * experiment.cl_unl.ls ** 4 * experiment.cl_unl.clpp / (2 * np.pi), 'k')

    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_1h_pos, color='b')
    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_1h_neg, color='b', ls='--')

    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_2h_pos, color='r')
    plt.plot(experiment.biases['ells'], scaling * cross_w_gals_2h_neg, color='r', ls='--')

    plt.yscale('log')
    plt.xlim([2, 3000])
    plt.ylim([1e-12,1e-7])
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.savefig('testing_crosswgals_{}bias_nZs{}_nMasses{}.png'.format(which_bias, nZs, nMasses))
    plt.show()
    plt.close()
    t1 = time.time()
    print('time taken:', t1-t0)


