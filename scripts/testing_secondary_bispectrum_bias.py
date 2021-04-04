# Script to generate secondary bispectrum bias on a cluster

import sys
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/lensing_rec_biases_code/')
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/')
import numpy as np
import matplotlib.pyplot as plt
from lensing_rec_biases_code import qest
from lensing_rec_biases_code import biases

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    which_bias = 'tsz' #'tsz' or 'cib'
    nlev_t = 18.  # uK arcmin
    beam_size = 1.  # arcmin
    lmax = 3000  # Maximum ell for the reconstruction

    # Initialise experiments with various different mass cuts
    SPT_5e15 = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir=5e15)

    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}  # Note that for now there is still cosmology dpendence in the cls defined within the experiment class

    nZs = 30
    nMasses = 30
    bin_width_out_second_bispec_bias = 30

    # Initialise a halo model object for the calculation, using mostly default parameters
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses)

    experiment = SPT_5e15

    hm_calc.get_tsz_bias(SPT_5e15, get_secondary_bispec_bias=True, bin_width_out_second_bispec_bias=bin_width_out_second_bispec_bias)

    # Save a dictionary with the bias we calculated to file
    experiment.save()

    # Plot the bias
    plt.figure(figsize=(5, 5))
    scaling = experiment.biases['second_bispec_bias_ells'] ** 4 / (2 * np.pi)

    convention_correction = 1 / (2 * np.pi)  # match FT convetion in QL

    plt.plot(experiment.biases['second_bispec_bias_ells'][experiment.biases[which_bias]['second_bispec']['1h'] > 0],
             (scaling * convention_correction * experiment.biases[which_bias]['second_bispec']['1h'])[
                 experiment.biases[which_bias]['second_bispec']['1h'] > 0], color='r',
             label=r'{}$^2-\kappa$, 1h'.format(which_bias), ls='--')
    plt.plot(experiment.biases['second_bispec_bias_ells'][experiment.biases[which_bias]['second_bispec']['1h'] < 0],
             -(scaling * convention_correction * experiment.biases[which_bias]['second_bispec']['1h'])[
                 experiment.biases[which_bias]['second_bispec']['1h'] < 0], color='r', ls='--')

    plt.xlim([2, 3000])

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    plt.savefig('testing_second_bispec_bias_nZs{}_nMasses{}_bin_width_out_second_bispec_bias{}.png'.format(nZs, nMasses, bin_width_out_second_bispec_bias))
    plt.close()


