# Script to generate secondary bispectrum bias on a cluster

import sys
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/lensing_rec_biases_code/')
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/')
import numpy as np
import matplotlib.pyplot as plt
from lensing_rec_biases_code import qest
from lensing_rec_biases_code import biases
import time

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t0 = time.time() # Time the run
    which_bias = 'tsz' #'tsz' or 'cib'
    lmax = 3500  # Maximum ell for the reconstruction
    nx = 256
    dx_arcmin = 1.0 * 2
    freq_GHz = np.array([27.3, 41.7, 93., 143., 225.,278.])  # [Hz]
    beam_size = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])  # [arcmin]
    nlev_t = np.array([52., 27., 5.8, 6.3, 15., 37.])  # [muK*arcmin]

    MV_ILC_bool = True

    # Set CIB halo model
    cib_model = 'planck13'  # 'vierro'
    # Initialise an experiment object. Store the list of params so we can later initialise it again within multiple processes
    massCut_Mvir=5e15
    experiment = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir,  nx, dx_arcmin, freq_GHz=freq_GHz, MV_ILC_bool=MV_ILC_bool)

    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}  # Note that for now there is still cosmology dpendence in the cls defined within the experiment class

    nZs = 3
    nMasses = 3
    bin_width_out_second_bispec_bias = 1000
    parallelise_secondbispec = False

    # Initialise a halo model object for the calculation, using mostly default parameters
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, cib_model=cib_model)

    if which_bias=='tsz':
        hm_calc.get_tsz_auto_biases(experiment, get_secondary_bispec_bias=True, \
                             bin_width_out_second_bispec_bias=bin_width_out_second_bispec_bias,
                                    parallelise_secondbispec=parallelise_secondbispec)
    elif which_bias=='cib':
        hm_calc.get_cib_auto_biases(experiment, get_secondary_bispec_bias=True, \
                             bin_width_out_second_bispec_bias=bin_width_out_second_bispec_bias,
                                    parallelise_secondbispec=parallelise_secondbispec)
    elif which_bias=='mixed':
        hm_calc.get_mixed_auto_biases(experiment, get_secondary_bispec_bias=True, \
                             bin_width_out_second_bispec_bias=bin_width_out_second_bispec_bias,
                                    parallelise_secondbispec=parallelise_secondbispec)

    # Save a dictionary with the bias we calculated to file
    experiment.save_biases()

    # Plot the bias
    plt.figure(figsize=(5, 5))
    scaling = experiment.biases['second_bispec_bias_ells'] ** 4 / (2 * np.pi)

    convention_correction = 1 # 1 when using QL #1 / (2 * np.pi)  # match FT convetion in QL

    plt.plot(experiment.biases['second_bispec_bias_ells'][experiment.biases[which_bias]['second_bispec']['1h'] > 0],
             (scaling * convention_correction * experiment.biases[which_bias]['second_bispec']['1h'])[
                 experiment.biases[which_bias]['second_bispec']['1h'] > 0], color='r',
             label=r'{}$^2-\kappa$, 1h'.format(which_bias), ls='--')
    plt.plot(experiment.biases['second_bispec_bias_ells'][experiment.biases[which_bias]['second_bispec']['1h'] < 0],
             -(scaling * convention_correction * experiment.biases[which_bias]['second_bispec']['1h'])[
                 experiment.biases[which_bias]['second_bispec']['1h'] < 0], color='r', ls='--')

    plt.yscale('log')
    plt.xlim([2, 3000])
    plt.ylim([1e-12, 1e-7])

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.show()
    plt.savefig('testing_second_bispec_bias_nZs{}_nMasses{}_which_bias{}_bin_width_out_second_bispec_bias{}.png'.format(nZs, nMasses, which_bias, bin_width_out_second_bispec_bias))
    plt.close()
    t1 = time.time()
    print('time taken:', t1-t0)


