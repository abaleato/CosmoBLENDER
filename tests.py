import matplotlib.pyplot as plt
import qest
import numpy as np
import hmvec as hm
import tools as tls
import biases


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
    # Note that for now there is still cosmology dependence in the cls defined within the experiment class
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}

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


def test_tsz_ps(hm_object):
    """ Check the tSZ power spectrum as a function of the pressure profile xmax out to which we integrate
        by generating plots that should resemble Fig.5 of Battaglia, Bond, Pfrommer & Sievers
            - Inputs:
                * hm_object = a biases.hm_framework object with the halo model information

    """
    nx_tsz = 10000 # FIXME: clarify what this is

    for xmax in range(6):
        oneHalo_ps_tz = np.zeros([nx_tsz, hm_object.nZs]) + 0j
        hm_object.hcos = hm.HaloModel(hm_object.hcos.zs, hm_object.hcos.ks, ms=hm_object.hcos.ms,
                                      mass_function='tinker', params=hm_object.cosmoParams, mdef='mean')
        hm_object.hcos.add_battaglia_pres_profile("y", family="pres", xmax=xmax, nxs=hm_object.nxs)

        for i, z in enumerate(hm_object.hcos.zs):
            # Temporary storage
            integrand_oneHalo_ps_tSZ = np.zeros([nx_tsz, hm_object.nMasses]) + 0j

            # M integral.
            for j, m in enumerate(hm_object.hcos.ms):
                # Make a map of the y field for this mass and z.
                y = tls.pkToPell(hm_object.hcos.comoving_radial_distance(hm_object.hcos.zs[i]), hm_object.hcos.ks,
                             hm_object.hcos.pk_profiles['y'][i, j] * (1 - np.exp(-(hm_object.hcos.ks /
                                                                                   hm_object.hcos.p['kstar_damping']))),
                             ellmax=nx_tsz - 1)
                # As a check compute the tSZ one halo power spectrum
                # hcos.nzm computes the HMF (See, e.g., eqn (1) of arXiv:1306.6721)
                integrand_oneHalo_ps_tSZ[:, j] = y * np.conjugate(y) * hm_object.hcos.nzm[i, j]

            oneHalo_ps_tz[:, i] = np.trapz(integrand_oneHalo_ps_tSZ, hm_object.hcos.ms, axis=-1)

        ps_oneHalo_tSZ = np.trapz(
            oneHalo_ps_tz * 1. / hm_object.hcos.comoving_radial_distance(hm_object.hcos.zs) ** 2 *\
            (hm_object.hcos.h_of_z(hm_object.hcos.zs)), hm_object.hcos.zs, axis=-1)  # *(1./hcos.comoving_radial_distance(hm_object.hcos.zs)**4*(hcos.h_of_z(hm_object.hcos.zs))**2),zs,axis=-1)

        plt.loglog(np.arange(len(ps_oneHalo_tSZ)),
                   np.arange(len(ps_oneHalo_tSZ)) * (np.arange(len(ps_oneHalo_tSZ)) + 1) / (2 * np.pi)
                   * tls.scale_sz(150.) ** 2 * hm_object.T_CMB ** 2 * ps_oneHalo_tSZ,
                   label='xmax={}, z>{}'.format(xmax, hm_object.z_min))
        print('completed xmax={}'.format(xmax))
    plt.xlim([200, 10000])
    plt.ylim([0.5, 10])
    plt.ylabel(r'$l(l+1)C_l/2\pi$')
    plt.xlabel(r'$l$')
    plt.legend()
    plt.grid(which='both')

if __name__ == '__main__':
    # Initialise the experiment and halo model object for which to run the tests

    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}
    # These give good results for the tSZ
    nMasses = 30
    nZs = 50
    z_min = 0.14  # 0.07
    z_max = 5
    m_min = 4.2e13  # 2e10
    m_max = 1e17
    k_min = 1e-3
    k_max = 10
    nks = 1001
    nxs = 30000
    hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, z_min=z_min, z_max=z_max,
                                  m_max=m_max, m_min=m_min, nxs=nxs, k_min=k_min, k_max=k_max, nks=nks)

    # Check the tSZ power spectrum
    test_tsz_ps(hm_calc)

    # Check CIB power spectrum
    #test_cib_ps()

    plt.show()

