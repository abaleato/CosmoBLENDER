import matplotlib.pyplot as plt
import biases
import qest


def test_cib_ps():
    """ Check the CIB power spectrum"""
    # Initialise experiment object
    nlev_t = 18.  # uK arcmin
    beam_size = 1.  # arcmin
    lmax = 3000  # Maximum ell for the reconstruction

    # Initialise experiments with various different mass cuts
    SPT_nocut = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir=5e17, freq_GHz=545.)

    # Initialise halo model calculator
    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}  # Note that for now there is still cosmology dpendence in the cls defined within the experiment class

    nZs = 60  # 30
    nMasses = 100  # 30
    z_max = 7

    # Initialise a halo model object for the CIB PS calculation, using mostly default parameters
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, z_max=z_max)

    clCIBCIB_oneHalo_ps, clCIBCIB_twoHalo_ps = hm_calc.get_cib_ps(SPT_nocut)
    plt.loglog(clCIBCIB_oneHalo_ps + clCIBCIB_twoHalo_ps, label='total')
    plt.loglog(clCIBCIB_oneHalo_ps, label='1 halo term')
    plt.loglog(clCIBCIB_twoHalo_ps, label='2 halo term')
    plt.loglog(clCIBCIB_oneHalo_ps + clCIBCIB_twoHalo_ps, label='total')
    plt.xlabel(r'l')
    plt.legend()
    plt.ylabel(r'$C_l$')
    plt.xlim([10, 1e4])
    return


if __name__ == '__main__':
    # Check CIB power spectrum
    test_cib_ps()
    plt.show()
