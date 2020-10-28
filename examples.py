import sys
sys.path.append('/Users/antonbaleatolizancos/Projects/lensing_rec_biases/lensing_rec_biases_code')
sys.path.insert(1,'/Users/antonbaleatolizancos/anaconda/envs/lensing_biases_python3_CIBbranch/lib/python3.8/site-packages/')

import numpy as np
import matplotlib.pyplot as plt
from lensing_rec_biases_code import tools as tls
from lensing_rec_biases_code import qest
from lensing_rec_biases_code import biases

#Initialise an experiment object
nlev_t = 18. # uK arcmin
beam_size = 1. #arcmin
lmax = 3000
SPT = qest.experiment(nlev_t, beam_size, lmax)

# Initialise a halo model object
# This should roughly match the cosmology in Nick's tSZ papers
cosmoParams = {'As':2.4667392631170437e-09,'ns':.96,'omch2':(0.25-.043)*.7**2,'ombh2':0.044*.7**2,'H0':70.}

# Initialise a halo model object for the calculation, using mostly default parameters
hm_calc = biases.hm_framework(cosmoParams=cosmoParams)

# Run the main function to obtain tsz biases
hm_calc.get_tsz_bias(SPT)

# Plot the biases
plt.figure(figsize=(5,5))
scaling = SPT.ls**4 /(2*np.pi) #Last division by pi is for delta(0)=A_sky/(2pi)^2=f_sky/pi

convention_correction = 1/(2*np.pi) #match FT convetion in QL

plt.plot(SPT.ls[SPT.biases['tsz']['trispec']['1h']>0],(scaling *convention_correction**2 * SPT.biases['tsz']['trispec']['1h'])[SPT.biases['tsz']['trispec']['1h']>0],color='b',label=r'tSZ 4pt, 1h ')

plt.plot(SPT.ls[SPT.biases['tsz']['trispec']['2h']>0],(scaling *convention_correction**2 * SPT.biases['tsz']['trispec']['2h'])[SPT.biases['tsz']['trispec']['2h']>0],color='b',label=r'tSZ 4pt, 2h ',ls='--')
plt.plot(SPT.ls[SPT.biases['tsz']['trispec']['2h']<0],-(scaling *convention_correction**2 * SPT.biases['tsz']['trispec']['2h'])[SPT.biases['tsz']['trispec']['2h']<0],color='b',ls='--')

plt.plot(SPT.ls[SPT.biases['tsz']['prim_bispec']['1h']>0],(scaling *convention_correction*SPT.biases['tsz']['prim_bispec']['1h'])[SPT.biases['tsz']['prim_bispec']['1h']>0],color='r',label=r'tSZ$^2-\kappa$, 1h')
plt.plot(SPT.ls[SPT.biases['tsz']['prim_bispec']['1h']<0],-(scaling *convention_correction*SPT.biases['tsz']['prim_bispec']['1h'])[SPT.biases['tsz']['prim_bispec']['1h']<0],color='r')

plt.plot(SPT.ls[SPT.biases['tsz']['prim_bispec']['2h']>0],(scaling *convention_correction*SPT.biases['tsz']['prim_bispec']['2h'])[SPT.biases['tsz']['prim_bispec']['2h']>0],color='r',label=r'tSZ$^2-\kappa$, 2h',ls='--')
plt.plot(SPT.ls[SPT.biases['tsz']['prim_bispec']['2h']<0],-(scaling *convention_correction*SPT.biases['tsz']['prim_bispec']['2h'])[SPT.biases['tsz']['prim_bispec']['2h']<0],color='r',ls='--')

plt.annotate(r'$0.05\,C_L^{\kappa \kappa}$', (2500,4e-11), rotation=-8)
plt.plot(SPT.ls, 0.05* SPT.ls**4 * SPT.clpp /(2*np.pi),'k')
plt.yscale('log')
plt.legend(fontsize=10, ncol=2)
plt.ylabel(r'$L^4 C_L^{\phi\phi}/ 2\pi$',fontsize=10)
plt.xlabel(r'$L$',fontsize=10)
plt.title(r'FFTlog, l$_{\mathrm{max}}=$'+str(lmax)+'$, M_{\mathrm{max}}=$'+'{:.2E}'.format(hm_calc.massCut), fontsize=10)
plt.xlim([2,3000])
plt.ylim([1e-11,1e-7])

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

#plt.savefig('../plots/tsz_bias_for_diff_mass_cuts/1D_reconstructions/biases_lmax{}_masscut{}.pdf'.format(lmax, '1e14'))

