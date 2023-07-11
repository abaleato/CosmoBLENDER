import matplotlib.pyplot as plt
import numpy as np
import hmvec as hm
from cosmoblender import qest
from cosmoblender import tools as tls
from cosmoblender import biases
import pyccl as ccl

def test_clbb_bias(hm_calc, exp):
    """ Check the biases to the power spectrum of delensed B-modes"""
    ells, cl_Btemp_x_Blens_bias, cl_Btemp_x_Btemp_bias, cl_Bdel_x_Bdel_bias = hm_calc.get_bias_to_delensed_clbb(exp)
    plt.loglog(ells, cl_Btemp_x_Blens_bias, label=r'Btemp_x_Blens')
    plt.loglog(ells, cl_Btemp_x_Btemp_bias, label=r'Btemp_x_Btemp')
    plt.loglog(ells, cl_Bdel_x_Bdel_bias, label=r'Bdel_x_Bdel')
    plt.ylabel(r'$C_l^{BB}$ [$\mu$K$^2$]')
    plt.xlabel(r'$l$')
    plt.legend()

def test_cib_ps(hm_object, exp_object, damp_1h_prof=True, cib_consistency=False):
    """ Check the CIB power spectrum"""
    # Shot noise to be added to compare to Planck data (or Websky)
    if int(exp_object.freq_GHz) == 353:
        shot_noise = 262 # see https://github.com/CLASS-SZ/notebooks/blob/main/class_sz_CIB_websky.ipynb
    elif int(exp_object.freq_GHz) == 217:
        shot_noise = 21 # see https://github.com/CLASS-SZ/notebooks/blob/main/class_sz_CIB_websky.ipynb
    elif int(exp_object.freq_GHz) == 545:
        shot_noise = 1690
    else:
        shot_noise = 0
        print('Assuming shot noise is zero!')


    # Our calculation
    clCIBCIB_oneHalo_ps, clCIBCIB_twoHalo_ps = hm_object.get_cib_ps(exp_object, damp_1h_prof=damp_1h_prof,
                                                                    cib_consistency=cib_consistency)
    # Compare to Yogesh' calculation
    ells = np.arange(3000)
    PII_yogesh_1h = hm_calc.hcos.get_power_1halo('cib', nu_obs=np.array([exp_object.freq_GHz * 1e9]))
    PII_yogesh_2h = hm_calc.hcos.get_power_2halo('cib', nu_obs=np.array([exp_object.freq_GHz * 1e9]))

    clCIBCIB_1h_yogesh = hm_calc.hcos.C_ii(ells, hm_object.hcos.zs, hm_object.hcos.ks, PII_yogesh_1h, dcdzflag=False)
    clCIBCIB_2h_yogesh = hm_calc.hcos.C_ii(ells, hm_object.hcos.zs, hm_object.hcos.ks, PII_yogesh_2h, dcdzflag=False)

    plt.loglog(tls.from_Jypersr_to_uK(exp_object.freq_GHz)**-2 * (clCIBCIB_oneHalo_ps + clCIBCIB_twoHalo_ps) + shot_noise, label=r'total, {}\,GHz'.format(exp_object.freq_GHz[0]), color='r')
    plt.loglog(tls.from_Jypersr_to_uK(exp_object.freq_GHz)**-2 * clCIBCIB_oneHalo_ps, label='1 halo term', color='g')
    plt.loglog(tls.from_Jypersr_to_uK(exp_object.freq_GHz)**-2 * clCIBCIB_twoHalo_ps, label='2 halo term', color='b')
    plt.loglog(ells, clCIBCIB_1h_yogesh + clCIBCIB_2h_yogesh + shot_noise, label=r'Hmvec tot, {}\,GHz'.format(exp_object.freq_GHz[0]), color='k', ls='-')
    plt.loglog(ells, clCIBCIB_1h_yogesh, label="Yogesh 1h", color='k', ls=':')
    plt.loglog(ells, clCIBCIB_2h_yogesh, label="Yogesh 2h", color='k', ls='--')

    plt.xlabel(r'l')
    plt.legend()
    plt.ylabel(r'$C_l\,[\mathrm{Jy}^2\,\mathrm{sr}^{-1}]$')
    plt.xlim([10, 1e4])
    plt.ylim([1,3e6])

    plt.title('CIB power spectrum')

def test_gal_cross_lensing(hm_object, exp_object, damp_1h_prof=False):
    """ Check the galaxy power spectrum from some HODs
        Note that the HODs ini pyccl and hmvec are not exactly the same, we 're just looking for order-of-magnitude
        agreement
    """

    lMmin = 10.5
    # PYCCL PART
    # First, set up the projection
    cosmo = ccl.CosmologyVanillaLCDM()
    z = np.linspace(0.01, 1, 256)
    nz = np.exp(-0.5 * ((z - 0.5) / 0.1) ** 2)
    b1 = np.ones_like(z)

    tracers = {}
    # Galaxy clustering
    tracers['g'] = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz), bias=(z, b1))
    # CMB lensing
    tracers['k'] = ccl.CMBLensingTracer(cosmo, z_source=1100)

    # Now get the 3D PS in the halo model of pyccl
    lk_arr = np.log(np.geomspace(1E-4, 100, 256))
    a_arr = 1. / (1 + np.linspace(0, 6., 40)[::-1])

    mass_def = ccl.halos.MassDef(200, 'critical')
    hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mass_def)
    cm = ccl.halos.ConcentrationDuffy08(mass_def)
    hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mass_def)

    profs = {}
    # This just defines which of these tracers should be normalized (e.g. overdensities)
    norm = {}

    # Galaxy clustering
    profs['g'] = ccl.halos.HaloProfileHOD(cm, lMmin_0=lMmin, siglM_0=np.sqrt(2)*0.2)
    norm['g'] = True
    # CMB lensing
    profs['k'] = ccl.halos.HaloProfileNFW(cm)
    norm['k'] = True

    tracer_list = list(profs.keys())
    profs2pt = {f'{t1}-{t2}': ccl.halos.Profile2pt() for t1 in tracer_list for t2 in tracer_list}
    profs2pt['g-g'] = ccl.halos.Profile2ptHOD()

    pks = {f'{t1}-{t2}': ccl.halos.halomod_Pk2D(cosmo, hmc,
                                                profs[t1],
                                                prof_2pt=profs2pt[f'{t1}-{t2}'],
                                                prof2=profs[t2],
                                                normprof1=norm[t1],
                                                normprof2=norm[t2],
                                                a_arr=a_arr,
                                                lk_arr=lk_arr)
           for t1 in tracer_list for t2 in tracer_list}

    ells = np.unique(np.geomspace(2, 2000, 128).astype(int)).astype(float)
    c_ells = {f'{t1}-{t2}': ccl.angular_cl(cosmo, tracers[t1], tracers[t2], ells, p_of_k_a=pks[f'{t1}-{t2}'])
              for t1 in tracer_list for t2 in tracer_list}

    plt.figure(figsize=(8, 4))
    for i1, t1 in enumerate(tracer_list):
        for t2 in tracer_list[i1:]:
            plt.plot(ells,  c_ells[f'{t1}-{t2}'] , label=f'{t1}-{t2}')
        plt.loglog()
        plt.legend(ncol=2)
        plt.xlabel(r'$\ell$', fontsize=14)
        plt.ylabel(r'$C_\ell$', fontsize=14)

    hod_name = "built-in"
    hm_object.hcos.add_hod(name=hod_name, mthresh=10 ** lMmin + hm_object.hcos.zs * 0.)
    ells = np.arange(3000)

    # OUR CALCULATION
    clkg_oneHalo_ps, clkg_twoHalo_ps = hm_object.get_g_cross_kappa(exp_object, hod_name, z, nz, damp_1h_prof=damp_1h_prof)
    plt.loglog(clkg_oneHalo_ps + clkg_twoHalo_ps, label='total', color='r')
    plt.loglog(clkg_oneHalo_ps, label='1 halo term', color='g')
    plt.loglog(clkg_twoHalo_ps, label='2 halo term', color='b')

    # HMVEC CALCULATION
    Pgm_hmvec_1h = hm_calc.hcos.get_power_1halo("nfw", hod_name)
    Pgm_hmvec_2h = hm_calc.hcos.get_power_2halo("nfw", hod_name)

    # TODO: Why is the 1-halo term more important than pyccl suggests?
    clgm_1h_hmvec = hm_calc.hcos.C_kg(ells, hm_object.hcos.zs, hm_object.hcos.ks, Pgm_hmvec_1h, gzs=z, gdndz=nz, lzs=1100.)
    clgm_2h_hmvec = hm_calc.hcos.C_kg(ells, hm_object.hcos.zs, hm_object.hcos.ks, Pgm_hmvec_2h, gzs=z, gdndz=nz, lzs=1100.)

    plt.loglog(ells, clgm_1h_hmvec + clgm_2h_hmvec, label="Hmvec tot", color='k', ls='-')
    plt.loglog(ells, clgm_1h_hmvec, label="Hmvec 1h", color='k', ls=':')
    plt.loglog(ells, clgm_2h_hmvec, label="Hmvec 2h", color='k', ls='--')

    plt.xlabel(r'l')
    plt.legend()
    #plt.ylabel(r'$C_l\,[\mathrm{Jy}^2\,\mathrm{sr}^{-1}]$')
    #plt.xlim([10, 1e4])
    #plt.ylim([1,3e6])

    plt.title('Galaxy - matter spectrum')

def test_tsz_ps(hm_object):
    """ Check the tSZ power spectrum as a function of the pressure profile xmax out to which we integrate
        by generating plots that should resemble Fig.5 of Battaglia, Bond, Pfrommer & Sievers
            - Inputs:
                * hm_object = a biases.hm_framework object with the halo model information

    """
    # TODO: this is outdated, use the version in biases.py
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

def test_tsz_bias(hm_object, experiment):
    hm_object.get_tsz_auto_biases(experiment, get_secondary_bispec_bias=False)
    plt.loglog(experiment.biases['ells'], experiment.biases['tsz']['trispec']['2h'])
    plt.loglog(experiment.biases['ells'], -experiment.biases['tsz']['trispec']['2h'], ls='--')

if __name__ == '__main__':
    which_test = 'test_CIB' # 'test_CIB' or 'test_tSZ' or 'test_clbb_bias' or 'test_gal_cross_lensing'

    # Initialise the experiment and halo model object for which to run the tests
    # You can specify a cosmological model -- in this case, match Websky
    H0 = 68.
    cosmoParams = {'As': 2.08e-9, 'ns': .965, 'omch2': (0.31 - 0.049) * (H0 / 100.) ** 2,
                   'ombh2': 0.049 * (H0 / 100.) ** 2, 'tau': 0.055, 'H0': H0}

    # This should roughly match the cosmology in Nick's tSZ papers
    #cosmoParams = {} #{'As': 2.4667392631170437e-09, 'ns': .96, 'omch2': (0.25 - .043) * .7 ** 2, 'ombh2': 0.044 * .7 ** 2, 'H0': 70.}

    if which_test == 'test_tsz_bias':
        # These give good results for the tSZ
        nMasses = 30
        nZs = 30
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

        # Foreground cleaning? Only relevant if many frequencies are provided
        MV_ILC_bool = False
        deproject_CIB = False
        deproject_tSZ = False
        fg_cleaning_dict = {'MV_ILC_bool': MV_ILC_bool, 'deproject_CIB': deproject_CIB, 'deproject_tSZ': deproject_tSZ}
        SPT_properties = {'nlev_t': np.array([18.]),
                          'beam_size': np.array([1.]),
                          'freq_GHz': np.array([150.])}
        # Initialise experiments with various different mass cuts
        SPT_nocut = qest.experiment(lmax=3000, massCut_Mvir=5e15, **SPT_properties, **fg_cleaning_dict)

        # Check the tSZ power spectrum
        test_tsz_bias(hm_calc, SPT_nocut)
        plt.show()

    if which_test == 'test_tSZ':
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
        plt.show()

    elif which_test == 'test_CIB':
        ''' A comparison of the CIB power spectrum against Fig 7 of the Websky paper'''

        # Foreground cleaning? Only relevant if many frequencies are provided
        MV_ILC_bool = False
        deproject_CIB = False
        deproject_tSZ = False
        fg_cleaning_dict = {'MV_ILC_bool': MV_ILC_bool, 'deproject_CIB': deproject_CIB, 'deproject_tSZ': deproject_tSZ}

        SPT_217_properties = {'nlev_t': np.array([18.]),
                          'beam_size': np.array([1.]),
                          'freq_GHz': np.array([217.])}
        SPT_353_properties = {'nlev_t': np.array([18.]),
                          'beam_size': np.array([1.]),
                          'freq_GHz': np.array([353.])}
        SPT_545_properties = {'nlev_t': np.array([18.]),
                          'beam_size': np.array([1.]),
                          'freq_GHz': np.array([545.])}

        # Initialise experiments with various different mass cuts
        SPT_217 = qest.experiment(lmax=10000, massCut_Mvir=3.5e15, **SPT_217_properties, **fg_cleaning_dict)
        SPT_353 = qest.experiment(lmax=10000, massCut_Mvir=3.5e15, **SPT_353_properties, **fg_cleaning_dict)
        SPT_545 = qest.experiment(lmax=10000, massCut_Mvir=3.5e15, **SPT_545_properties, **fg_cleaning_dict)

        # Initialise a halo model object for the CIB PS calculation, using mostly default parameters
        nZs = 30  # 30
        nMasses = 30  # 30
        z_max = 3
        k_max = 10
        k_min = 1e-4
        cib_model = 'planck13'

        hm_calc = biases.hm_framework(m_min=1e8, cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, z_max=z_max,
                                      k_max=k_max, k_min=k_min, cib_model=cib_model)
        # Check CIB power spectrum
        test_cib_ps(hm_calc, SPT_217, damp_1h_prof=True, cib_consistency=False)
        test_cib_ps(hm_calc, SPT_353, damp_1h_prof=True, cib_consistency=False)
        test_cib_ps(hm_calc, SPT_545, damp_1h_prof=True, cib_consistency=False)

        plt.show()

    elif which_test == 'test_gal_cross_lensing':
        # Foreground cleaning? Only relevant if many frequencies are provided
        MV_ILC_bool = True
        deproject_CIB = False
        deproject_tSZ = False
        fg_cleaning_dict = {'MV_ILC_bool': MV_ILC_bool, 'deproject_CIB': deproject_CIB, 'deproject_tSZ': deproject_tSZ}

        SPT_properties = {'nlev_t': np.array([18.]),
                          'beam_size': np.array([1.]),
                          'freq_GHz': np.array([545.])}

        # Initialise experiments with various different mass cuts
        SPT_5e15 = qest.experiment(lmax=3000, massCut_Mvir=5e15, **SPT_properties, **fg_cleaning_dict)

        # Initialise a halo model object for the CIB PS calculation, using mostly default parameters
        nZs = 20  # 30
        nMasses = 20  # 30
        z_max = 3
        cib_model = 'viero'
        hm_calc = biases.hm_framework(m_min=1e10/0.7, cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, z_max=z_max,
                                      cib_model=cib_model)
        # Check CIB power spectrum
        test_gal_cross_lensing(hm_calc, SPT_5e15, damp_1h_prof=True)
        plt.show()

    elif which_test=='test_clbb_bias':
        # Initialise experiment object
        nlev_t = 18.  # uK arcmin
        beam_size = 1.  # arcmin
        lmax = 3000  # Maximum ell for the reconstruction
        freq_GHz = 143.
        massCut_Mvir = 5e17
        # Initialise experiments with various different mass cuts
        SPT_nocut = qest.experiment(nlev_t, beam_size, lmax, massCut_Mvir=massCut_Mvir, freq_GHz=freq_GHz)
        # Initialise a halo model object for the CIB PS calculation, using mostly default parameters
        nZs = 30  # 30
        nMasses = 30  # 30
        z_max = 4
        hm_calc = biases.hm_framework(cosmoParams=cosmoParams, nZs=nZs, nMasses=nMasses, z_max=z_max)
        # Check CIB power spectrum
        test_clbb_bias(hm_calc, SPT_nocut)
        plt.show()

