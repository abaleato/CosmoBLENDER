import matplotlib.pyplot as plt
import numpy as np
import biases
import qest
import tools as tls


def test_cib_ps():
    """ Check the CIB power spectrum"""
    # Initialise experiment object
    nlev_t = 18.  # uK arcmin
    beam_size = 1.  # arcmin
    lmax = 3000  # Maximum ell for the reconstruction
    lmax_out = 3000  # Maximum ell for the output

    # Initialise experiments with various different mass cuts
    SPT_nocut = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir=5e17, freq_GHz=857.)

    # Initialise halo model calculator
    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,'H0': 70.}  # Note that for now there is still cosmology dependence in the cls defined within the experiment class

    nZs = 30  # 30
    nMasses = 30  # 30
    z_max = 6
    z_min = 0.01
    m_min=1e6
    m_max=1e15

    # Initialise a halo model object for the CIB PS calculation, using mostly default parameters
    hm_calc = biases.hm_framework(lmax_out=lmax_out, cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, z_max=z_max, z_min=z_min, m_min=m_min, m_max=m_max)

    clCIBCIB_oneHalo_ps, clCIBCIB_twoHalo_ps = hm_calc.get_cib_ps(SPT_nocut)
    plt.loglog(clCIBCIB_oneHalo_ps + clCIBCIB_twoHalo_ps, label='total')
    plt.loglog(clCIBCIB_oneHalo_ps, label='1 halo term')
    plt.loglog(clCIBCIB_twoHalo_ps, label='2 halo term')
    plt.loglog(clCIBCIB_oneHalo_ps + clCIBCIB_twoHalo_ps, label='total')
    plt.xlabel(r'l')
    plt.legend()
    plt.ylabel(r'$C_l$')
    plt.xlim([10, 1e4])
    plt.title('CIB power spectrum')
    return

def test_tSZ_ps():
    """ Check the tSZ power spectrum. Produces a plot that can be compared to, e.g., Figure 5 of Battaglia, Bond, Pfrommer & Sievers"""
    # Initialise experiment object
    nlev_t = 18.  # uK arcmin
    beam_size = 1.  # arcmin
    lmax = 3000  # Maximum ell for the reconstruction
    lmax_out = 10000  # Maximum ell for the output

    # Initialise experiments with various different mass cuts
    SPT_nocut = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir=5e17, freq_GHz=857.)

    # Initialise halo model calculator
    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,'H0': 70.}  # Note that for now there is still cosmology dependence in the cls defined within the experiment class

    nZs = 30  # 30
    nMasses = 30  # 30
    z_max = 6
    z_min = 0.01
    m_min=1e6
    m_max=1e15

    # Initialise a halo model object for the tSZ PS calculation, using mostly default parameters
    hm_calc = biases.hm_framework(lmax_out=lmax_out, cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, z_max=z_max, z_min=z_min, m_min=m_min, m_max=m_max)

    cltSZtSZ_oneHalo_ps, cltSZtSZ_twoHalo_ps = hm_calc.get_tsz_ps(SPT_nocut)
    plt.loglog(np.arange(len(cltSZtSZ_oneHalo_ps)),np.arange(len(cltSZtSZ_oneHalo_ps))*(np.arange(len(cltSZtSZ_oneHalo_ps))+1)/(2*np.pi)*tls.scale_sz(150.)**2 * cltSZtSZ_oneHalo_ps,label='z>{}'.format(z_min))

    plt.xlim([200, 10000])
    plt.ylim([0.5, 10])
    plt.ylabel(r'$l(l+1)C_l/2\pi$')
    plt.xlabel(r'$l$')
    plt.legend()
    plt.grid(which='both')
    plt.title('tSZ power spectrum at 150GHz')
    return


if __name__ == '__main__':
    # Check CIB power spectrum
    test_cib_ps()
    plt.figure()
    # Check tSZ power spectrum
    #test_tSZ_ps()
    plt.show()
