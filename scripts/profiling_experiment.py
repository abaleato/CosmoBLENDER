import numpy as np
from cosmoblender import qest
from cosmoblender import biases

# Foreground cleaning? Only relevant if many frequencies are provided
MV_ILC_bool = True
deproject_CIB = False
deproject_tSZ = False
fg_cleaning_dict = {'MV_ILC_bool':MV_ILC_bool, 'deproject_CIB':deproject_CIB, 'deproject_tSZ':deproject_tSZ}

SPT_properties = {'nlev_t': np.array([18.]),
                  'beam_size':np.array([1.]),
                  'freq_GHz': np.array([143.]), 'nx_secbispec':128}

# Initialise experiments with various different mass cuts
SPT_5e15 = qest.experiment(lmax = 3000, massCut_Mvir=5e15, **SPT_properties, **fg_cleaning_dict)

SO_properties = {'nlev_t': np.array([52., 27., 5.8, 6.3, 15., 37.]),
                 'beam_size':np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]),
                 'freq_GHz': np.array([27.3, 41.7, 93., 143., 225.,278.]), 'nx_secbispec':128}

# Initialise experiments with various different mass cuts
SO_5e15 = qest.experiment(lmax = 3000, massCut_Mvir=5e15, **SO_properties, **fg_cleaning_dict)

# Choose an experiment
experiment = SPT_5e15

# You can specify a cosmological model -- in this case, match Websky
H0 = 68.
cosmoParams = {'As':2.08e-9,'ns':.965,'omch2':(0.31-0.049)*(H0/100.)**2,'ombh2':0.049*(H0/100.)**2,'tau':0.055,'H0':H0}

z_max = 3 #3
nZs = 20 #50
nMasses = 20  #30
Mmin = 1e8 #Keep this low -- the 2h term of the bispectrum bias can be sensitive to quite low-mass halos

# Set CIB halo model
cib_model='planck13'#'vierro'

# Initialise a halo model object for the calculation, using mostly default parameters
hm_calc = biases.hm_framework(cosmoParams=cosmoParams, m_min=Mmin, nZs=nZs, nMasses=nMasses, cib_model=cib_model, z_max=z_max)

which_bias = 'tsz' # 'tsz' or 'cib' or 'mixed'

# Calculate secondary bispectrum bias? Note that this is a bit lower that the other implemented terms
get_secondary_bispec_bias = True
parallelise_secondbispec = False

if which_bias=='tsz':
    hm_calc.get_tsz_auto_biases(experiment, get_secondary_bispec_bias=get_secondary_bispec_bias,
                                parallelise_secondbispec=parallelise_secondbispec, bin_width_out_second_bispec_bias=100)
if which_bias=='cib':
    hm_calc.get_cib_auto_biases(experiment, get_secondary_bispec_bias=get_secondary_bispec_bias,
                               parallelise_secondbispec=parallelise_secondbispec, bin_width_out_second_bispec_bias=100)
if which_bias=='mixed':
    hm_calc.get_mixed_auto_biases(experiment, get_secondary_bispec_bias=get_secondary_bispec_bias,
                                 parallelise_secondbispec=parallelise_secondbispec, bin_width_out_second_bispec_bias=100)



