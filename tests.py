import matplotlib.pyplot as plt
import biases
import qest
import numpy as np
import hmvec as hm
import tools as tls


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


def test_tsz_ps():
    """
    Check the tSZ power spectrum by generating plots that should resemble
    Fig.5 of Battaglia, Bond, Pfrommer & Sievers
    """
    T_CMB = 2.7255e6
    # This should roughly match the cosmology in Nick's tSZ papers
    cosmoParams = {'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2,
                   'H0': 70.}

    # Setup hmvec. This is the range of zs and masses
    nMasses = 30
    nZs = 50
    z_min = 0.14  # 0.07
    zs = np.linspace(z_min, 5, nZs)
    M_min = 4.2e13  # 2e10
    ms = np.geomspace(M_min, 1e17, nMasses)  # masses
    ks = np.geomspace(1e-3, 10, 1001)  # wavenumbers

    nx_tsz = 10000

    for xmax in range(6):
        oneHalo_ps_tz = np.zeros([nx_tsz, nZs]) + 0j
        hcos = hm.HaloModel(zs, ks, ms=ms, mass_function='tinker', params=cosmoParams, mdef='mean')
        hcos.add_battaglia_pres_profile("y", family="pres", xmax=xmax, nxs=30000)

        for i, z in enumerate(hcos.zs):
            # Temporary storage
            integrand_oneHalo_ps_tSZ = np.zeros([nx_tsz, nMasses]) + 0j

            # M integral.
            for j, m in enumerate(hcos.ms):
                # Make a map of the y field for this mass and z.
                y = tls.pkToPell(hcos.comoving_radial_distance(zs[i]), ks,
                             hcos.pk_profiles['y'][i, j] * (1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                             ellmax=nx_tsz - 1)
                # As a check compute the tSZ one halo power spectrum
                # hcos.nzm computes the HMF (See, e.g., eqn (1) of arXiv:1306.6721)
                integrand_oneHalo_ps_tSZ[:, j] = y * np.conjugate(y) * hcos.nzm[i, j]

            oneHalo_ps_tz[:, i] = np.trapz(integrand_oneHalo_ps_tSZ, hcos.ms, axis=-1)

        ps_oneHalo_tSZ = np.trapz(
            oneHalo_ps_tz * 1. / hcos.comoving_radial_distance(hcos.zs) ** 2 * (hcos.h_of_z(hcos.zs)), zs,
            axis=-1)  # *(1./hcos.comoving_radial_distance(hcos.zs)**4*(hcos.h_of_z(hcos.zs))**2),zs,axis=-1)

        plt.loglog(np.arange(len(ps_oneHalo_tSZ)),
                   np.arange(len(ps_oneHalo_tSZ)) * (np.arange(len(ps_oneHalo_tSZ)) + 1) / (2 * np.pi) * tls.scale_sz(
                       150.) ** 2 * T_CMB ** 2 * ps_oneHalo_tSZ, label='xmax={}, z>{}'.format(xmax, z_min))
        print('completed xmax={}'.format(xmax))
    plt.xlim([200, 10000])
    plt.ylim([0.5, 10])
    plt.ylabel(r'$l(l+1)C_l/2\pi$')
    plt.xlabel(r'$l$')
    plt.legend()
    plt.grid(which='both')

if __name__ == '__main__':
    # Check CIB power spectrum
    #test_cib_ps()

    # Check the tSZ power spectrum
    test_tsz_ps()
    plt.show()
