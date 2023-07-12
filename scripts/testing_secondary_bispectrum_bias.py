# Script to generate secondary bispectrum bias on a cluster
'''
import sys
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/lensing_rec_biases_code/')
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases/')
from lensing_rec_biases_code import qest
from lensing_rec_biases_code import biases
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from cosmoblender import qest
from cosmoblender import biases

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    t0 = time.time() # Time the run
    which_bias = 'tsz' #'tsz' or 'cib'
    lmax = 3000  # Maximum ell for the reconstruction
    nx_secbispec = 256
    dx_arcmin_secbispec = 1.0

    # Foreground cleaning? Only relevant if many frequencies are provided
    MV_ILC_bool = False
    deproject_CIB = False
    deproject_tSZ = False
    fg_cleaning_dict = {'MV_ILC_bool': MV_ILC_bool, 'deproject_CIB': deproject_CIB, 'deproject_tSZ': deproject_tSZ}
    SPT_properties = {'nlev_t': np.array([18.]), 'beam_size': np.array([1.]), 'freq_GHz': np.array([150.]),
                      'dx_arcmin_secbispec':dx_arcmin_secbispec, 'nx_secbispec':nx_secbispec}
    # Initialise experiments with various different mass cuts
    experiment = qest.experiment(lmax=lmax, massCut_Mvir=5e15, **SPT_properties, **fg_cleaning_dict)

    # Set CIB halo model
    cib_model = 'planck13'  # 'vierro'
    # You can specify a cosmological model -- in this case, match Websky
    H0 = 68.
    cosmoParams = {'As': 2.08e-9, 'ns': .965, 'omch2': (0.31 - 0.049) * (H0 / 100.) ** 2,
                   'ombh2': 0.049 * (H0 / 100.) ** 2, 'tau': 0.055, 'H0': H0}

    nZs = 20
    nMasses = 20
    bin_width_out_second_bispec_bias = 100
    parallelise_secondbispec = False
    max_workers = None # Force serial for now

    # Initialise a halo model object for the calculation, using mostly default parameters
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, cib_model=cib_model)

    if which_bias=='tsz':
        hm_calc.get_tsz_auto_biases(experiment, get_secondary_bispec_bias=True, \
                             bin_width_out_second_bispec_bias=bin_width_out_second_bispec_bias,
                                    parallelise_secondbispec=parallelise_secondbispec, max_workers=max_workers)
    elif which_bias=='cib':
        hm_calc.get_cib_auto_biases(experiment, get_secondary_bispec_bias=True, \
                             bin_width_out_second_bispec_bias=bin_width_out_second_bispec_bias,
                                    parallelise_secondbispec=parallelise_secondbispec, max_workers=max_workers)
    elif which_bias=='mixed':
        hm_calc.get_mixed_auto_biases(experiment, get_secondary_bispec_bias=True, \
                             bin_width_out_second_bispec_bias=bin_width_out_second_bispec_bias,
                                    parallelise_secondbispec=parallelise_secondbispec, max_workers=max_workers)

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
    plt.savefig('testing_second_bispec_bias_nZs{}_nMasses{}_which_bias{}_bin_width_out_second_bispec_bias{}.png'.format(nZs, nMasses, which_bias, bin_width_out_second_bispec_bias))
    plt.close()
    t1 = time.time()
    print('time taken:', t1-t0)


