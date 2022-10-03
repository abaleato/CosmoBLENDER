import sys
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases')

import numpy as np
import matplotlib.pyplot as plt
from lensing_rec_biases_code import tools as tls
from lensing_rec_biases_code import qest
from lensing_rec_biases_code import biases
plt.style.use('/Users/antonbaleatolizancos/Documents/Science/plt_styles/paper.mplstyle')

lmax = 3000 # Maximum ell for the reconstruction

'''
freq_GHz = np.array([143.])#217.
nlev_t = np.array([18.]) # uK arcmin
beam_size = np.array([1.]) #arcmin
'''

freq_GHz = np.array([27.3, 41.7, 93., 143., 225., 278.])   # [Hz]#np.array([27.e9, 39.e9, 93.e9, 145.e9, 225.e9, 280.e9])   # [Hz] #np.array([27.3e9, 41.7e9, 93.e9, 143.e9, 225.e9, 278.e9])   # [Hz]
beam_size = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])   # [arcmin]
nlev_t = np.array([52., 27., 5.8, 6.3, 15., 37.])  # [muK*arcmin]

MV_ILC_bool = True

# Initialise experiments with various different mass cuts
SO_5e15 = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir=5e15, freq_GHz=freq_GHz, MV_ILC_bool=MV_ILC_bool)

# This should roughly match the cosmology in Nick's tSZ papers
cosmoParams = {'As':2.4667392631170437e-09,'ns':.96,'omch2':(0.25-.043)*.7**2,'ombh2':0.044*.7**2,'H0':70.} #Note that for now there is still cosmology dpendence in the cls defined within the experiment class

z_max = 3 #3
nZs = 30 #50
nMasses = 30 #30

# Set CIB halo model
cib_model='planck13'#'vierro'

# Initialise a halo model object for the calculation, using mostly default parameters
hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, cib_model=cib_model, z_max=z_max)

hm_calc.get_tsz_auto_biases(SO_5e15)

which_bias = 'tsz' # 'tsz' or 'cib' or 'mixed'
experiment = SO_5e15

# Break down contributions into n-halo terms?
breakdown = False

get_secondary_bispec_bias = False

plt.figure()
title = ''

scaling = experiment.biases['ells'] ** 4 / 4.
'''
if get_secondary_bispec_bias:
    # TODO: a factor of 1/2 would make the 2ndary bispec agree better with expectations
    scaling_second_bispec_bias = experiment_2ndarybispec.biases['second_bispec_bias_ells'] ** 4 / 4.
    scaling_second_bispec_bias[:3] = np.nan
'''
# Split into negative and positive parts for plotting convenience
prim_bispec_1h_pos, prim_bispec_1h_neg = tls.split_positive_negative(experiment.biases[which_bias]['prim_bispec']['1h'])
prim_bispec_2h_pos, prim_bispec_2h_neg = tls.split_positive_negative(experiment.biases[which_bias]['prim_bispec']['2h'])
prim_bispec_tot_pos, prim_bispec_tot_neg = tls.split_positive_negative(
    experiment.biases[which_bias]['prim_bispec']['2h'] + experiment.biases[which_bias]['prim_bispec']['1h'])

trispec_1h_pos, trispec_1h_neg = tls.split_positive_negative(experiment.biases[which_bias]['trispec']['1h'])
trispec_2h_pos, trispec_2h_neg = tls.split_positive_negative(experiment.biases[which_bias]['trispec']['2h'])
trispec_tot_pos, trispec_tot_neg = tls.split_positive_negative(
    experiment.biases[which_bias]['trispec']['2h'] + experiment.biases[which_bias]['trispec']['1h'])
'''
if get_secondary_bispec_bias:
    sec_bispec_1h_pos, sec_bispec_1h_neg = tls.split_positive_negative(
        experiment_2ndarybispec.biases[which_bias]['second_bispec']['1h'])
    # FIXME: for now, ony 1h term for secondary bispec bias
    sec_bispec_tot_pos, sec_bispec_tot_neg = tls.split_positive_negative(
        0 + experiment_2ndarybispec.biases[which_bias]['second_bispec']['1h'])
'''
plt.plot(experiment.biases['ells'], scaling * trispec_tot_pos, color='b',
         label=r'{} trispec. bias, tot '.format(which_bias))
plt.plot(experiment.biases['ells'], scaling * trispec_tot_neg, color='b', ls='--')

if breakdown:
    title = '_withbreakdown'
    plt.plot(experiment.biases['ells'], scaling * trispec_1h_pos, color='b',
             label=r'{} trispec. bias, 1h '.format(which_bias), ls=':')
    plt.plot(experiment.biases['ells'], scaling * trispec_1h_neg, color='b', ls=':')

    plt.plot(experiment.biases['ells'], scaling * trispec_2h_pos, color='b',
             label=r'{} trispec. bias, 2h '.format(which_bias), ls='-.')
    plt.plot(experiment.biases['ells'], scaling * trispec_2h_neg, color='b', ls='-.')

plt.plot(experiment.biases['ells'], scaling * prim_bispec_tot_pos, color='r',
         label=r'{} prim. bispec. bias, tot'.format(which_bias))
plt.plot(experiment.biases['ells'], scaling * prim_bispec_tot_neg, color='r', ls='--')

if breakdown:
    plt.plot(experiment.biases['ells'], scaling * prim_bispec_1h_pos, color='r',
             label=r'{} prim. bispec. bias, 1h '.format(which_bias), ls=':')
    plt.plot(experiment.biases['ells'], scaling * prim_bispec_1h_neg, color='r', ls=':')

    plt.plot(experiment.biases['ells'], scaling * prim_bispec_2h_pos, color='r',
             label=r'{} prim. bispec. bias, 2h '.format(which_bias), ls='-.')
    plt.plot(experiment.biases['ells'], scaling * prim_bispec_2h_neg, color='r', ls='-.')
'''
if get_secondary_bispec_bias:
    plt.plot(experiment_2ndarybispec.biases['second_bispec_bias_ells'], scaling_second_bispec_bias * sec_bispec_tot_pos,
             color='orange', label=r'{} sec. bispec. bias, tot'.format(which_bias))
    plt.plot(experiment_2ndarybispec.biases['second_bispec_bias_ells'], scaling_second_bispec_bias * sec_bispec_tot_neg,
             color='orange', ls='--')

if breakdown:
    plt.plot(experiment.biases['second_bispec_bias_ells'], scaling_second_bispec_bias * sec_bispec_1h_pos,
             color='orange', label=r'{} sec. bispec. bias, 1h '.format(which_bias), ls=':')
    plt.plot(experiment.biases['second_bispec_bias_ells'], scaling_second_bispec_bias * sec_bispec_1h_neg,
             color='orange', ls=':')
'''
plt.plot(experiment.cl_unl.ls, 0.05 * experiment.cl_unl.ls ** 4 * experiment.cl_unl.clpp / 4., 'k')
plt.yscale('log')

plt.ylabel(r'$C_L^{\kappa\kappa}$')
plt.xlabel(r'$L$')
plt.xlim([100, 3000])
#plt.ylim([1e-11, 1e-7])

plt.show()