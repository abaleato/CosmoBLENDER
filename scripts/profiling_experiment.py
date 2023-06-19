import sys
sys.path.insert(0,'/Users/antonbaleatolizancos/Projects/lensing_rec_biases')

import numpy as np
import matplotlib.pyplot as plt
from lensing_rec_biases_code import qest
plt.style.use('/Users/antonbaleatolizancos/Documents/Science/plt_styles/paper.mplstyle')

lmax = 3000 # Maximum ell for the reconstruction

freq_GHz = np.array([27.3, 41.7, 93., 143., 225., 278.])   # [Hz]#np.array([27.e9, 39.e9, 93.e9, 145.e9, 225.e9, 280.e9])   # [Hz] #np.array([27.3e9, 41.7e9, 93.e9, 143.e9, 225.e9, 278.e9])   # [Hz]
beam_size = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])   # [arcmin]
nlev_t = np.array([52., 27., 5.8, 6.3, 15., 37.])  # [muK*arcmin]

MV_ILC_bool = True

# Initialise experiments with various different mass cuts
SO_5e15 = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir=5e15, freq_GHz=freq_GHz, MV_ILC_bool=MV_ILC_bool)

