'''
Terminology:
itgnd = integrand
oneH = one halo
twoH = two halo
'''

import numpy as np
import hmvec as hm
from . import tools as tls
from . import qest
from . import second_bispec_bias_stuff as sbbs
import quicklens as ql
import concurrent

class Hm_minimal:
    """ A helper class to encapsulate some essential attributes of hm_framework() objects to be passed to parallelized
        workers, saving as much memory as possible
        - Inputs:
            * hm_full = an instance of the hm_framework() class
    """
    def __init__(self, hm_full):
        dict = {"m_consistency": hm_full.m_consistency, "ms": hm_full.hcos.ms, "nzm": hm_full.hcos.nzm,
                "ms_rescaled": hm_full.ms_rescaled, "zs": hm_full.hcos.zs, "ks": hm_full.hcos.ks,
                "comoving_radial_distance": hm_full.hcos.comoving_radial_distance(hm_full.hcos.zs),
                "uk_profiles": hm_full.hcos.uk_profiles, "pk_profiles": hm_full.hcos.pk_profiles,
                "p": hm_full.hcos.p, "bh_ofM": hm_full.hcos.bh_ofM, "lmax_out": hm_full.lmax_out,
                "y_consistency": hm_full.y_consistency, "g_consistency": hm_full.g_consistency,
                "I_consistency": hm_full.I_consistency, "Pzk": hm_full.hcos.Pzk, "nMasses": hm_full.nMasses,
                "nZs": hm_full.nZs, "hods":hm_full.hcos.hods, "CIB_satellite_filter":hm_full.CIB_satellite_filter,
                "CIB_central_filter":hm_full.CIB_central_filter}
        self.__dict__ = dict

class hm_framework:
    """ Set the halo model parameters """
    def __init__(self, lmax_out=3000, m_min=1e10, m_max=5e15, nMasses=30, z_min=0.07, z_max=3, nZs=30, k_min = 1e-4,\
                 k_max=10, nks=1001, mass_function='sheth-torman', mdef='vir', cib_model='planck13', cosmoParams=None
                 , xmax=5, nxs=40000):
        """ Inputs:
                * lmax_out = int. Maximum multipole at which to return the lensing reconstruction
                * m_min = Minimum virial mass for the halo model calculation
                * m_max = Maximum virial mass for the halo model calculation (bot that massCut_Mvir will overide this)
                * nMasses = Integer. Number of steps in mass for the integrals
                * z_min = Minimum redshift for the halo model calc
                * z_max = Maximum redshift for the halo model calc
                * nZs = Integer. Number of steps in redshift for the integrals
                * k_min = Minimum k for the halo model calc
                * k_max = Maximum k for the halo model calc
                * nks = Integer. Number of steps in k for the integrals
                * mass_function = String. Halo mass function to use. Must be coded into hmvec
                * mdef = String. Mass definition. Must be defined in hmvec for the chosen mass_function
                * cib_model = CIB halo model and fit params. Either 'planck13' or 'viero' (after Viero et al 13.)
                * cosmoParams = Dictionary of cosmological parameters to initialised HaloModel hmvec object
                * xmax = Float. Electron pressure profile integral xmax (see further docs at hmvec.add_nfw_profile() )
                * nxs = Integer. Electron pressure profile integral number of x's
        """
        self.lmax_out = lmax_out
        self.nMasses = nMasses
        self.m_min = m_min
        self.m_max = m_max
        self.z_min = z_min
        self.z_max = z_max
        self.nZs = nZs
        self.mass_function = mass_function
        self.mdef = mdef
        zs = np.linspace(z_min,z_max,nZs) # redshifts
        ms = np.geomspace(m_min,m_max,nMasses) # masses
        ks = np.geomspace(k_min,k_max,nks) # wavenumbers
        self.T_CMB = 2.7255e6
        self.nZs = nZs
        self.nxs = nxs
        self.xmax = xmax
        self.cosmoParams = cosmoParams

        self.hcos = hm.HaloModel(zs,ks,ms=ms,mass_function=mass_function,params=cosmoParams,mdef=mdef)
        self.hcos.add_battaglia_pres_profile("y",family="pres",xmax=xmax,nxs=nxs)
        self.hcos.set_cibParams(cib_model)

        self.ms_rescaled = self.hcos.ms[...]/self.hcos.rho_matter_z(0)

        self.m_consistency = np.zeros(len(self.hcos.zs))
        self.y_consistency = np.zeros(len(self.hcos.zs))
        self.g_consistency = np.zeros(len(self.hcos.zs))
        self.I_consistency = np.zeros(len(self.hcos.zs))

        self.CIB_central_filter = None
        self.CIB_satellite_filter = None

    def __str__(self):
        """ Print out halo model calculator properties """
        m_min = '{:.2e}'.format(self.m_min)
        m_max = '{:.2e}'.format(self.m_max)
        z_min = '{:.2f}'.format(self.z_min)
        z_max = '{:.2f}'.format(self.z_max)

        return 'M_min: ' + m_min + '  M_max: ' + m_max + '  n_Masses: '+ str(self.nMasses) + '\n'\
               + '  z_min: ' + z_min + '  z_max: ' + z_max + '  n_zs: ' + str(self.nZs) +  '\n'\
               +'  Mass function: ' + self.mass_function + '  Mass definition: ' + self.mdef

    def get_matter_consistency(self, exp):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over dark matter
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut<self.hcos.ms]=0
        I = np.trapz(self.hcos.nzm*self.hcos.bh_ofM*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)
        self.m_consistency = 1 - I # A function of z

    def get_galaxy_consistency(self, exp, survey_name, lmax_proj=None):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over galaxy number density
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        if lmax_proj is None:
            lmax_proj = self.lmax_out
        ugal_proj_of_Mlow = np.zeros((len(self.hcos.zs), lmax_proj+1))
        for i, z in enumerate(self.hcos.zs):
            ugal_proj_of_Mlow[i, :] = tls.pkToPell(self.hcos.comoving_radial_distance(self.hcos.zs[i]),
                                        self.hcos.ks, self.hcos.uk_profiles['nfw'][i, 0],
                                        ellmax=lmax_proj) # A function of z and k
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut<self.hcos.ms]=0
        I = np.trapz(self.hcos.nzm*self.hcos.bh_ofM*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)
        W_of_Mlow = (self.hcos.hods[survey_name]['Nc'][:, 0] + self.hcos.hods[survey_name]['Ns'][:, 0])[:,None]\
                    / self.hcos.hods[survey_name]['ngal'][:,None] * ugal_proj_of_Mlow # A function of z and k
        self.g_consistency = ((1 - I)/(self.hcos.ms[0]/self.hcos.rho_matter_z(0)))[:,None]*W_of_Mlow #Function of z & k

    def get_tsz_consistency(self, exp, lmax_proj=None):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over tsz emission.
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        if lmax_proj is None:
            lmax_proj = self.lmax_out

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        tsz_filter = exp.get_tsz_filter()

        W_of_Mlow = np.zeros((len(self.hcos.zs), lmax_proj + 1))
        for i, z in enumerate(self.hcos.zs):
            W_of_Mlow[i, :] = tsz_filter * tls.pkToPell(self.hcos.comoving_radial_distance(self.hcos.zs[i]),
                                                   self.hcos.ks, self.hcos.pk_profiles['y'][i, 0],
                                                   ellmax=lmax_proj)  # A function of z and k
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut < self.hcos.ms] = 0
        I = np.trapz(self.hcos.nzm * self.hcos.bh_ofM * self.hcos.ms / self.hcos.rho_matter_z(0) * mMask, self.hcos.ms,
                     axis=-1)
        self.y_consistency = ((1 - I) / (self.hcos.ms[0] / self.hcos.rho_matter_z(0)))[:,None] * W_of_Mlow

    def get_cib_consistency(self, exp, lmax_proj=None):
        """
        Calculate consistency relation for 2-halo term given some mass cut for an integral over CIB emission.
        Variable names are roughly inspired by Appendix A of Mead et al 2020
        Input:
            * exp = a qest.experiment object
        """
        if lmax_proj is None:
            lmax_proj = self.lmax_out
        self.get_CIB_filters(exp)
        ucen_plus_usat_of_Mlow = np.zeros((len(self.hcos.zs), lmax_proj+1))
        for i, z in enumerate(self.hcos.zs):
            u_of_Mlow_proj = tls.pkToPell(self.hcos.comoving_radial_distance(self.hcos.zs[i]),
                                        self.hcos.ks, self.hcos.uk_profiles['nfw'][i, 0, :], ellmax=lmax_proj)
            u_cen = self.CIB_central_filter[:, i, 0]
            u_sat = self.CIB_satellite_filter[:, i, 0] * u_of_Mlow_proj
            ucen_plus_usat_of_Mlow[i, :] = u_cen + u_sat # A function of z and k
        mMask = np.ones(self.nMasses)
        mMask[exp.massCut<self.hcos.ms]=0
        I = np.trapz(self.hcos.nzm*self.hcos.bh_ofM*self.hcos.ms/self.hcos.rho_matter_z(0)*mMask,self.hcos.ms, axis=-1)
        W_of_Mlow =  ucen_plus_usat_of_Mlow # A function of z and k
        self.I_consistency = ((1 - I)/(self.hcos.ms[0]/self.hcos.rho_matter_z(0)))[:,None]*W_of_Mlow # Function of z & k

    def get_tsz_auto_biases(self, exp, fftlog_way=True, get_secondary_bispec_bias=False, bin_width_out=30, \
                     bin_width_out_second_bispec_bias=250, parallelise_secondbispec=True, damp_1h_prof=True,
                     tsz_consistency=False, max_workers=None):
        """
        Calculate the tsz biases to the CMB lensing auto-spectrum (C^{\phi\phi}_L)
        given an "experiment" object (defined in qest.py).
        Uses serialization/pickling to parallelize the calculation of each z-point in the integrands
        Input:
            * exp = a qest.experiment object
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
            * (optional) get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) bin_width_out_second_bispec_bias = int. Bin width of the output secondary bispectrum bias
            * (optional) parallelise_secondbispec = bool.
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) tsz_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                  low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
        """
        hcos = self.hcos
        self.get_matter_consistency(exp)
        if tsz_consistency:
            self.get_tsz_consistency(exp, lmax_proj=exp.lmax)

        # Output ells
        ells_out = np.linspace(1, self.lmax_out) #np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        #nx = self.lmax_out + 1 if fftlog_way else exp.pix.nx
        nx = len(ells_out) if fftlog_way else exp.pix.nx

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        exp.tsz_filter = exp.get_tsz_filter()

        # The one and two halo bias terms -- these store the itgnd to be integrated over z.
        # Dimensions depend on method
        oneH_4pt = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        twoH_1_3 = oneH_4pt.copy(); twoH_2_2 = oneH_4pt.copy(); oneH_cross = oneH_4pt.copy();
        twoH_cross = oneH_4pt.copy();

        lbins_sec_bispec_bias = np.arange(10, self.lmax_out + 1, bin_width_out_second_bispec_bias)
        # Get QE normalisation at required multipole bins for secondary bispec bias calculation
        exp.qe_norm_at_lbins_sec_bispec = exp.qe_norm.get_ml(lbins_sec_bispec_bias).specs['cl']
        L_array_sec_bispec_bias = exp.qe_norm.get_ml(lbins_sec_bispec_bias).ls
        oneH_second_bispec = np.zeros([len(L_array_sec_bispec_bias), self.nZs]) + 0j
        twoH_second_bispec = np.zeros([len(L_array_sec_bispec_bias), self.nZs]) + 0j

        # If using FFTLog, we can compress the normalization to 1D
        if fftlog_way:
            norm_bin_width = 40  # These are somewhat arbitrary
            lmin = 1  # These are somewhat arbitrary
            lbins = np.arange(lmin, exp.lmax, norm_bin_width)
            exp.qe_norm_compressed = np.interp(ells_out, exp.qe_norm.get_ml(lbins).ls, exp.qe_norm.get_ml(lbins).specs['cl'])
        else:
            exp.qe_norm_compressed = exp.qe_norm

        # Run in parallel
        print('Launching parallel processes...')
        hm_minimal = Hm_minimal(self)
        exp_minimal = qest.Exp_minimal(exp)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            n = len(hcos.zs)
            outputs = executor.map(tsZ_auto_itgrnds_each_z, np.arange(n), n * [ells_out], n * [fftlog_way],
                                   n * [get_secondary_bispec_bias], n * [parallelise_secondbispec], n * [damp_1h_prof],
                                   n * [L_array_sec_bispec_bias], n * [exp_minimal], n * [hm_minimal])

            for idx, itgnds_at_i in enumerate(outputs):
                oneH_4pt[...,idx], oneH_cross[...,idx], twoH_2_2[...,idx], twoH_1_3[...,idx], twoH_cross[...,idx],\
                oneH_second_bispec[...,idx], twoH_second_bispec[...,idx] = itgnds_at_i

        # Convert the NFW profile in the cross bias from kappa to phi
        conversion_factor = np.nan_to_num(1 / (0.5 * ells_out*(ells_out+1) )) if fftlog_way else ql.spec.cl2cfft(np.nan_to_num(1 / (0.5 * np.arange(self.lmax_out+1)*(np.arange(self.lmax_out+1)+1) )),exp.pix).fft

        # Integrate over z
        yyyy_trispec_intgrnd = self.T_CMB**4 * tls.limber_itgrnd_kernel(hcos, 4) * tls.y_window(hcos)**4
        kyy_bispec_intgrnd = 2 * self.T_CMB**2 * tls.limber_itgrnd_kernel(hcos, 3) * tls.my_lensing_window(hcos, 1100.)\
                             * tls.y_window(hcos)**2

        exp.biases['tsz']['trispec']['1h'] = np.trapz(oneH_4pt * yyyy_trispec_intgrnd, hcos.zs,axis=-1)
        exp.biases['tsz']['trispec']['2h'] = np.trapz((twoH_1_3 + twoH_2_2)*yyyy_trispec_intgrnd, hcos.zs, axis=-1)
        exp.biases['tsz']['prim_bispec']['1h'] = conversion_factor * np.trapz(oneH_cross * kyy_bispec_intgrnd,
                                                                              hcos.zs,axis=-1)
        exp.biases['tsz']['prim_bispec']['2h'] = conversion_factor * np.trapz(twoH_cross * kyy_bispec_intgrnd,
                                                                              hcos.zs,axis=-1)
        if get_secondary_bispec_bias:
            # Perm factors implemented in the get_secondary_bispec_bias_at_L() function
            # TODO: Why doesn't this use a conversion_factor?
            # TODO: does this give the correct permutation factor of 4?
            exp.biases['tsz']['second_bispec']['1h'] = 2 * np.trapz( oneH_second_bispec * kyy_bispec_intgrnd,
                                                                       hcos.zs, axis=-1)
            exp.biases['tsz']['second_bispec']['2h'] = 2 * np.trapz(twoH_second_bispec * kyy_bispec_intgrnd,
                                                                    hcos.zs, axis=-1)
            exp.biases['second_bispec_bias_ells'] = L_array_sec_bispec_bias

        if fftlog_way:
            exp.biases['ells'] = ells_out
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['tsz']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return

    def get_tsz_cross_biases(self, exp, gzs, gdndz, fftlog_way=True, bin_width_out=30, survey_name='LSST',
                             damp_1h_prof=True, gal_consistency=False, tsz_consistency=False, max_workers=None):
        """
        Calculate the tsz biases to the cross-correlation of CMB lensing with a galaxy survey, (C^{g\phi}_L)
        given an "experiment" object (defined in qest.py)
        Uses serialization/pickling to parallelize the calculation of each z-point in the integrands
        Input:
            * exp = a qest.experiment object
            * gzs = array. Redsfhits at which the dndz is defined. Assumed to be zero otherwise.
            * gdndz = array of same size as gzs. The dndz of the galaxy sample (does not need to be normalized)
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) survey_name = str. Name labelling the HOD characterizing the survey we are x-ing lensing with
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) gal_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
        """
        hcos = self.hcos
        if tsz_consistency:
            self.get_tsz_consistency(exp, lmax_proj=exp.lmax)
        if gal_consistency:
            self.get_galaxy_consistency(exp, survey_name)

        # Output ells
        ells_out = np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.pix.nx

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        exp.tsz_filter = exp.get_tsz_filter()

        # The one and two halo bias terms -- these store the itgnd to be integrated over z.
        # Dimensions depend on method
        oneH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        twoH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j

        # If using FFTLog, we can compress the normalization to 1D
        if fftlog_way:
            norm_bin_width = 40  # These are somewhat arbitrary
            lmin = 1  # These are somewhat arbitrary
            lbins = np.arange(lmin, exp.lmax, norm_bin_width)
            exp.qe_norm_compressed = np.interp(ells_out, exp.qe_norm.get_ml(lbins).ls, exp.qe_norm.get_ml(lbins).specs['cl'])
        else:
            exp.qe_norm_compressed = exp.qe_norm

        # Run in parallel
        print('Launching parallel processes...')
        hm_minimal = Hm_minimal(self)
        exp_minimal = qest.Exp_minimal(exp)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            n = len(hcos.zs)
            outputs = executor.map(tsZ_cross_itgrnds_each_z, np.arange(n), n * [ells_out], n * [fftlog_way],
                                   n * [damp_1h_prof], n * [exp_minimal], n * [hm_minimal], n * [survey_name])

            for idx, itgnds_at_i in enumerate(outputs):
                oneH_cross[...,idx], twoH_cross[...,idx] = itgnds_at_i

        # Integrate over z
        gyy_intgrnd = self.T_CMB ** 2 * tls.limber_itgrnd_kernel(hcos, 3) \
                      * tls.gal_window(hcos, hcos.zs, gzs, gdndz) * tls.y_window(hcos) ** 2
        exp.biases['tsz']['cross_w_gals']['1h'] = np.trapz(oneH_cross * gyy_intgrnd, hcos.zs, axis=-1)
        exp.biases['tsz']['cross_w_gals']['2h'] = np.trapz(twoH_cross * gyy_intgrnd, hcos.zs, axis=-1)
        if fftlog_way:
            exp.biases['ells'] = ells_out
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['tsz']['cross_w_gals']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['cross_w_gals']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['cross_w_gals']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['cross_w_gals']['2h']).get_ml(lbins).specs['cl']
            return

    def get_tsz_ps(self, exp, damp_1h_prof=True, max_workers=None):
        """
        Calculate the tSZ power spectrum
        Input:
            * exp = a qest.experiment object
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        """
        hcos = self.hcos
        nx = self.lmax_out+1

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_ps_tz = np.zeros([nx,self.nZs])+0j

        # Run in parallel
        print('Launching parallel processes...')
        hm_minimal = Hm_minimal(self)
        exp_minimal = qest.Exp_minimal(exp)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            n = len(hcos.zs)
            outputs = executor.map(tsZ_ps_itgrnds_each_z, np.arange(n), n * [damp_1h_prof], n * [exp_minimal], n * [hm_minimal])

            for idx, itgnds_at_i in enumerate(outputs):
                oneH_ps_tz[...,idx] = itgnds_at_i

        # Integrate over z
        yy_integrand = self.T_CMB ** 2 * tls.limber_itgrnd_kernel(hcos, 2) * tls.y_window(hcos) ** 2
        ps_oneH_tSZ = np.trapz( oneH_ps_tz * yy_integrand, hcos.zs, axis=-1)

        # ToDo: implement 2 halo term for tSZ PS tests
        ps_twoH_tSZ = np.zeros(ps_oneH_tSZ.shape)
        return ps_oneH_tSZ, ps_twoH_tSZ

    def get_g_cross_kappa(self, exp, survey_name, gzs, gdndz, damp_1h_prof=True, fftlog_way=True, gal_consistency=False):
        """
        Calculate galaxy cross CMB lensing spectrum. This is a for a test.
        Input:
            * exp = a qest.experiment object
            * survey_name = str. Name of the HOD
            * gzs = array of floats. Redshifts at which the dndz is defined
            * gdndz = array of floats. dndz of the galaxy sample, at the zs given by gzs. Need not be normalized
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
            * (optional) gal_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
        """
        hcos = self.hcos
        if gal_consistency:
            self.get_galaxy_consistency(exp, survey_name)
        self.get_matter_consistency(exp)

        # Output ells
        ells_out = np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)

        nx = self.lmax_out+1

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_ps = np.zeros([nx,self.nZs])+0j; twoH_ps = oneH_ps.copy()
        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_ps = np.zeros([nx,self.nMasses])+0j
            itgnd_2h_1g = itgnd_1h_ps.copy(); itgnd_2h_1m = itgnd_1h_ps.copy()

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks, hcos.uk_profiles['nfw'][i,j], ellmax=self.lmax_out)
                gal = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks, hcos.uk_profiles['nfw'][i, j], ellmax=self.lmax_out)
                # TODO: should ngal in denominator depend on z? ms_rescaled doesn't
                galfft = gal / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal,exp.pix).fft / \
                                                                                         hcos.hods[survey_name]['ngal'][
                                                                                             i]
                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]

                if damp_1h_prof:
                    gal_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                       hcos.ks, hcos.uk_profiles['nfw'][i, j]
                                       *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                    kap_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                            hcos.uk_profiles['nfw'][i, j] *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                            ellmax=self.lmax_out)
                    galfft_damp = gal_damp / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal,
                                                                                                        exp.pix).fft / \
                                                                                        hcos.hods[survey_name]['ngal'][
                                                                                            i]
                    kfft_damp = kap_damp * self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp,
                                                                                                  exp.pix).fft * \
                                                                                  self.ms_rescaled[j]
                else:
                    kap_damp = kap; gal_damp = gal; kfft_damp = kfft

                mean_Ngal = hcos.hods[survey_name]['Nc'][i, j] + hcos.hods[survey_name]['Ns'][i, j]
                # Accumulate the itgnds
                itgnd_1h_ps[:,j] = mean_Ngal * galfft_damp * np.conjugate(kfft_damp)*hcos.nzm[i,j]
                # TODO: Implement 2h including consistency
                itgnd_2h_1g[:, j] = mean_Ngal * np.conjugate(galfft)*hcos.nzm[i,j]*hcos.bh_ofM[i,j]
                itgnd_2h_1m[:, j] = np.conjugate(kfft)*hcos.nzm[i,j]*hcos.bh_ofM[i,j]

            # Perform the m integrals
            oneH_ps[:,i]=np.trapz(itgnd_1h_ps,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            twoH_ps[:, i] = (np.trapz(itgnd_2h_1g, hcos.ms, axis=-1) + self.g_consistency[i])\
                            * (np.trapz(itgnd_2h_1m, hcos.ms, axis=-1) + self.m_consistency[i]) * pk
        # Integrate over z
        gk_intgrnd = tls.limber_itgrnd_kernel(hcos, 2) * tls.gal_window(hcos, hcos.zs, gzs, gdndz)\
                     * tls.my_lensing_window(hcos, 1100.)
        ps_oneH = np.trapz( oneH_ps * gk_intgrnd, hcos.zs, axis=-1)
        ps_twoH = np.trapz( twoH_ps * gk_intgrnd, hcos.zs, axis=-1)
        return ps_oneH, ps_twoH

    def get_CIB_filters(self, exp):
        """
        Get f_cen and f_sat factors for CIB halo model scaled by foreground cleaning weights. That is,
        compute \Sum_{\nu} f^{\nu}(z,M) w^{\nu, ILC}_l. While you're at it, convert to CMB units.
        Input:
            * exp = a qest.experiment object
        """
        if len(exp.freq_GHz)>1:
            f_cen_array = np.zeros((len(exp.freq_GHz), len(self.hcos.zs), len(self.hcos.ms)))
            f_sat_array = f_cen_array.copy()
            for i, freq in enumerate(np.array(exp.freq_GHz*1e9)):
                freq = np.array([freq])
                f_cen_array[i, :, :] = tls.from_Jypersr_to_uK(freq[0]*1e-9) * self.hcos.get_fcen(freq)[:,:,0]
                f_sat_array[i, :, :] = tls.from_Jypersr_to_uK(freq[0]*1e-9)\
                                       * self.hcos.get_fsat(freq, cibinteg='trap', satmf='Tinker')[:,:,0]
            # Compute \Sum_{\nu} f^{\nu}(z,M) w^{\nu, ILC}_l
            self.CIB_central_filter = np.sum(exp.ILC_weights[:,:,None,None] * f_cen_array, axis=1)
            self.CIB_satellite_filter = np.sum(exp.ILC_weights[:,:,None,None] * f_sat_array, axis=1)
        else:
            # Single-frequency scenario. Return two (nZs, nMs) array containing f_cen(M,z) and f_sat(M,z)
            # Compute \Sum_{\nu} f^{\nu}(z,M) w^{\nu, ILC}_l
            self.CIB_central_filter = tls.from_Jypersr_to_uK(exp.freq_GHz)\
                                      * self.hcos.get_fcen(exp.freq_GHz*1e9)[:,:,0][np.newaxis,:,:]
            self.CIB_satellite_filter = tls.from_Jypersr_to_uK(exp.freq_GHz) \
                                        * self.hcos.get_fsat(exp.freq_GHz*1e9, cibinteg='trap',
                                                             satmf='Tinker')[:,:,0][np.newaxis,:,:]

    def get_cib_auto_biases(self, exp, fftlog_way=True, get_secondary_bispec_bias=False, bin_width_out=30, \
                     bin_width_out_second_bispec_bias=250, parallelise_secondbispec=True, damp_1h_prof=True,
                     cib_consistency=False, max_workers=None):
        """
        Calculate the CIB biases to the CMB lensing auto-spectrum (C^{\phi\phi}_L)
        given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) bin_width_out_second_bispec_bias = int. Bin width of the output secondary bispectrum bias
            * (optional) parallelise_secondbispec = bool.
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) cib_consistency = Bool. Whether to impose consistency condition on CIB to correct for missing
                                          low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
        """
        hcos = self.hcos

        # Pre-calculate consistency for 2h integrals
        self.get_matter_consistency(exp)
        if cib_consistency:
            self.get_cib_consistency(exp, lmax_proj=exp.lmax)

        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)

        # Output ells
        ells_out = np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)
        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        IIII_1h = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        IIII_2h_1_3 = IIII_1h.copy(); IIII_2h_2_2 = IIII_1h.copy(); oneH_cross = IIII_1h.copy();
        twoH_cross = IIII_1h.copy()

        # TODO: choose an Lmin that makes sense given Limber
        lbins_sec_bispec_bias = np.arange(10, self.lmax_out + 1, bin_width_out_second_bispec_bias)
        # Get QE normalisation at required multipole bins for secondary bispec bias calculation
        exp.qe_norm_at_lbins_sec_bispec = exp.qe_norm.get_ml(lbins_sec_bispec_bias).specs['cl']
        L_array_sec_bispec_bias = exp.qe_norm.get_ml(lbins_sec_bispec_bias).ls
        oneH_second_bispec = np.zeros([len(L_array_sec_bispec_bias), self.nZs]) + 0j
        twoH_second_bispec = np.zeros([len(L_array_sec_bispec_bias), self.nZs]) + 0j

        # If using FFTLog, we can compress the normalization to 1D
        if fftlog_way:
            norm_bin_width = 40  # These are somewhat arbitrary
            lmin = 1  # These are somewhat arbitrary
            lbins = np.arange(lmin, exp.lmax, norm_bin_width)
            exp.qe_norm_compressed = np.interp(ells_out, exp.qe_norm.get_ml(lbins).ls, exp.qe_norm.get_ml(lbins).specs['cl'])
        else:
            exp.qe_norm_compressed = exp.qe_norm

        # Run in parallel
        print('Launching parallel processes...')
        hm_minimal = Hm_minimal(self)
        exp_minimal = qest.Exp_minimal(exp)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            n = len(hcos.zs)
            outputs = executor.map(cib_auto_itgrnds_each_z, np.arange(n), n * [ells_out], n * [fftlog_way],
                                   n * [get_secondary_bispec_bias], n * [parallelise_secondbispec], n * [damp_1h_prof],
                                   n * [L_array_sec_bispec_bias], n * [exp_minimal], n * [hm_minimal])

            for idx, itgnds_at_i in enumerate(outputs):
                IIII_1h[...,idx], oneH_cross[...,idx], IIII_2h_2_2[...,idx], IIII_2h_1_3[...,idx], twoH_cross[...,idx],\
                oneH_second_bispec[...,idx], twoH_second_bispec[...,idx] = itgnds_at_i

        # Convert the NFW profile in the cross bias from kappa to phi (bc the QEs give phi)
        conversion_factor = np.nan_to_num(1 / (0.5 * ells_out*(ells_out+1) )) if fftlog_way else ql.spec.cl2cfft(np.nan_to_num(1 / (0.5 * np.arange(self.lmax_out+1)*(np.arange(self.lmax_out+1)+1) )),exp.pix).fft

        # itgnd factors from Limber projection
        # kII_itgnd has a perm factor of 2
        IIII_itgnd = tls.limber_itgrnd_kernel(hcos, 4) * tls.CIB_window(hcos)**4
        kII_itgnd = 2 * tls.limber_itgrnd_kernel(hcos, 3) * tls.my_lensing_window(hcos, 1100.) * tls.CIB_window(hcos)**2

        # Integrate over z
        exp.biases['cib']['trispec']['1h'] = np.trapz( IIII_itgnd*IIII_1h, hcos.zs, axis=-1)
        exp.biases['cib']['trispec']['2h'] = np.trapz( IIII_itgnd*(IIII_2h_1_3 + IIII_2h_2_2), hcos.zs, axis=-1)
        exp.biases['cib']['prim_bispec']['1h'] = conversion_factor * np.trapz( oneH_cross*kII_itgnd,
                                                                               hcos.zs, axis=-1)
        exp.biases['cib']['prim_bispec']['2h'] = conversion_factor * np.trapz( twoH_cross*kII_itgnd,
                                                                               hcos.zs, axis=-1)

        if get_secondary_bispec_bias:
            # Perm factors implemented in the get_secondary_bispec_bias_at_L() function
            exp.biases['cib']['second_bispec']['1h'] = 2 * np.trapz( oneH_second_bispec * kII_itgnd, hcos.zs, axis=-1)
            exp.biases['cib']['second_bispec']['2h'] = 2 * np.trapz( twoH_second_bispec * kII_itgnd, hcos.zs, axis=-1)
            exp.biases['second_bispec_bias_ells'] = L_array_sec_bispec_bias

        if fftlog_way:
            exp.biases['ells'] = ells_out
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['cib']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return

    def get_cib_cross_biases(self, exp, gzs, gdndz, fftlog_way=True, bin_width_out=30, survey_name='LSST',
                             damp_1h_prof=True, gal_consistency=False, cib_consistency=False, max_workers=None):
        """
        Calculate the CIB biases to the cross-correlation of CMB lensing with a galaxy survey, (C^{g\phi}_L)
        given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * gzs = array. Redsfhits at which the dndz is defined. Assumed to be zero otherwise.
            * gdndz = array of same size as gzs. The dndz of the galaxy sample (does not need to be normalized)
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) survey_name = str. Name labelling the HOD characterizing the survey we are x-ing lensing with
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) gal_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
        """
        hcos = self.hcos

        if gal_consistency:
            self.get_galaxy_consistency(exp, survey_name)
        if cib_consistency:
            self.get_cib_consistency(exp, lmax_proj=exp.lmax)

        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)

        # Output ells
        ells_out = np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j;
        twoH_cross = oneH_cross.copy()

        # If using FFTLog, we can compress the normalization to 1D
        if fftlog_way:
            norm_bin_width = 40  # These are somewhat arbitrary
            lmin = 1  # These are somewhat arbitrary
            lbins = np.arange(lmin, exp.lmax, norm_bin_width)
            exp.qe_norm_compressed = np.interp(ells_out, exp.qe_norm.get_ml(lbins).ls, exp.qe_norm.get_ml(lbins).specs['cl'])
        else:
            exp.qe_norm_compressed = exp.qe_norm

        # Run in parallel
        print('Launching parallel processes...')
        hm_minimal = Hm_minimal(self)
        exp_minimal = qest.Exp_minimal(exp)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            n = len(hcos.zs)
            outputs = executor.map(cib_cross_itgrnds_each_z, np.arange(n), n * [ells_out], n * [fftlog_way],
                                   n * [damp_1h_prof], n * [exp_minimal], n * [hm_minimal], n * [survey_name])

            for idx, itgnds_at_i in enumerate(outputs):
                oneH_cross[...,idx], twoH_cross[...,idx] = itgnds_at_i

        # itgnd factors from Limber projection (adapted to hmvec conventions)
        gII_itgnd = tls.limber_itgrnd_kernel(hcos, 3) * tls.gal_window(hcos, hcos.zs, gzs, gdndz) * tls.CIB_window(hcos)**2

        # Integrate over z
        exp.biases['cib']['cross_w_gals']['1h'] = np.trapz( oneH_cross*gII_itgnd, hcos.zs, axis=-1)
        exp.biases['cib']['cross_w_gals']['2h'] = np.trapz( twoH_cross*gII_itgnd, hcos.zs, axis=-1)

        if fftlog_way:
            exp.biases['ells'] = ells_out
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['cib']['cross_w_gals']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['cross_w_gals']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['cross_w_gals']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['cross_w_gals']['2h']).get_ml(lbins).specs['cl']
            return

    def get_cib_ps(self, exp, damp_1h_prof=True, cib_consistency=False):
        """
        Calculate the CIB power spectrum.

        Note that this uses a consistency relation for the 2h term, while typical
        implementations do not. We must therefore use halo model parameters obtained after fitting to the data a
        model that does include the consistency term

        Input:
            * exp = a qest.experiment object
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) cib_consistency = Bool. Whether to enforce consistency condition correcting for lack of low mass
                                    halos in truncated integral. When using hmvec/Planck/Viero best fit params, must
                                    be False.
        """
        hcos = self.hcos
        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)
        # Compute consistency relation for 2h term
        if cib_consistency:
            self.get_cib_consistency(exp)

        nx = exp.lmax+1

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_ps = np.zeros([nx,self.nZs])+0j
        twoH_ps = np.zeros([nx,self.nZs])+0j
        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_ps = np.zeros([nx,self.nMasses])+0j
            itgnd_2h_1g = np.zeros([nx, self.nMasses]) + 0j

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                # project the galaxy profiles
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.uk_profiles['nfw'][i, j],
                                 ellmax=exp.lmax)

                u_cen = self.CIB_central_filter[:,i,j]
                u_sat = self.CIB_satellite_filter[:,i,j] * u

                if damp_1h_prof:
                    u_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                          hcos.uk_profiles['nfw'][i, j]
                                          * (1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=exp.lmax)
                    u_sat_damp = self.CIB_satellite_filter[:, i, j] * u_damp
                else:
                    u_sat_damp=u_sat

                # Accumulate the itgnds
                itgnd_1h_ps[:, j] = hcos.nzm[i, j] * (2*u_cen*u_sat_damp + u_sat_damp*u_sat_damp)
                itgnd_2h_1g[:, j] = hcos.nzm[i, j] * hcos.bh_ofM[i, j] * (u_cen + u_sat)

                # Perform the m integrals
            oneH_ps[:, i] = np.trapz(itgnd_1h_ps, hcos.ms, axis=-1)
            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=exp.lmax)
            twoH_ps[:, i] = (np.trapz(itgnd_2h_1g, hcos.ms, axis=-1) +self.I_consistency[i])** 2 * pk

        # Integrate over z
        II_itgrnd = tls.limber_itgrnd_kernel(hcos, 2) * tls.CIB_window(hcos) ** 2
        clCIBCIB_oneH_ps = np.trapz( oneH_ps * II_itgrnd, hcos.zs, axis=-1)
        clCIBCIB_twoH_ps = np.trapz( twoH_ps * II_itgrnd, hcos.zs, axis=-1)
        return clCIBCIB_oneH_ps, clCIBCIB_twoH_ps

    def get_mixed_auto_biases(self, exp, fftlog_way=True, get_secondary_bispec_bias=False, bin_width_out=30, \
                         bin_width_out_second_bispec_bias=250, parallelise_secondbispec=True, damp_1h_prof=True,
                         tsz_consistency=False, cib_consistency=False, max_workers=None):
        """
        Calculate biases to the CMB lensing auto-spectrum (C^{\phi\phi}_L) from both CIB and tSZ
        given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) bin_width_out_second_bispec_bias = int. Bin width of the output secondary bispectrum bias
            * (optional) parallelise_secondbispec = bool.
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) tsz_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) cib_consistency = Bool. Whether to impose consistency condition on CIB to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
        """
        hcos = self.hcos
        # Get consistency conditions for 2h terms
        self.get_matter_consistency(exp)
        if tsz_consistency:
            self.get_tsz_consistency(exp, lmax_proj=exp.lmax)
        if cib_consistency:
            self.get_cib_consistency(exp, lmax_proj=exp.lmax)

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        exp.tsz_filter = exp.get_tsz_filter()
        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)

        # Output ells
        ells_out = np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        Iyyy_1h = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        IIyy_1h = Iyyy_1h.copy(); yIII_1h = Iyyy_1h.copy(); Iyyy_2h_2_2 = Iyyy_1h.copy(); Iyyy_2h_1_3 = Iyyy_1h.copy();
        IIyy_2h_2_2 = Iyyy_1h.copy(); IIyy_2h_1_3 = Iyyy_1h.copy(); yIII_2h_1_3 = Iyyy_1h.copy();
        yIII_2h_2_2 = Iyyy_1h.copy(); oneH_cross = Iyyy_1h.copy(); twoH_cross = Iyyy_1h.copy();
        IyIy_2h_2_2 = Iyyy_1h.copy(); IyIy_2h_1_3 = Iyyy_1h.copy(); IyIy_1h = Iyyy_1h.copy()

        # TODO: choose an Lmin that makes sense given Limber
        lbins_sec_bispec_bias = np.arange(10, self.lmax_out + 1, bin_width_out_second_bispec_bias)
        # Get QE normalisation at required multipole bins for secondary bispec bias calculation
        exp.qe_norm_at_lbins_sec_bispec = exp.qe_norm.get_ml(lbins_sec_bispec_bias).specs['cl']
        L_array_sec_bispec_bias = exp.qe_norm.get_ml(lbins_sec_bispec_bias).ls
        oneH_second_bispec = np.zeros([len(L_array_sec_bispec_bias), self.nZs]) + 0j
        twoH_second_bispec = np.zeros([len(L_array_sec_bispec_bias), self.nZs]) + 0j

        # If using FFTLog, we can compress the normalization to 1D
        if fftlog_way:
            norm_bin_width = 40  # These are somewhat arbitrary
            lmin = 1  # These are somewhat arbitrary
            lbins = np.arange(lmin, exp.lmax, norm_bin_width)
            exp.qe_norm_compressed = np.interp(ells_out, exp.qe_norm.get_ml(lbins).ls, exp.qe_norm.get_ml(lbins).specs['cl'])
        else:
            exp.qe_norm_compressed = exp.qe_norm

        # Run in parallel
        print('Launching parallel processes...')
        hm_minimal = Hm_minimal(self)
        exp_minimal = qest.Exp_minimal(exp)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            n = len(hcos.zs)
            outputs = executor.map(mixed_auto_itgrnds_each_z, np.arange(n), n * [ells_out], n * [fftlog_way],
                                   n * [get_secondary_bispec_bias], n * [parallelise_secondbispec], n * [damp_1h_prof],
                                   n * [L_array_sec_bispec_bias], n * [exp_minimal], n * [hm_minimal])

            for idx, itgnds_at_i in enumerate(outputs):
                Iyyy_1h[...,idx], IIyy_1h[...,idx], IyIy_1h[...,idx], yIII_1h[...,idx], oneH_cross[...,idx], \
                oneH_second_bispec[...,idx], twoH_second_bispec[...,idx], Iyyy_2h_1_3[...,idx], IIyy_2h_1_3[...,idx], IyIy_2h_1_3[...,idx], \
                yIII_2h_1_3[...,idx], Iyyy_2h_2_2[...,idx], IIyy_2h_2_2[...,idx], IyIy_2h_2_2[...,idx], \
                yIII_2h_2_2[...,idx], twoH_cross[...,idx]  = itgnds_at_i

        # Convert the NFW profile in the cross bias from kappa to phi
        conversion_factor = np.nan_to_num(1 / (0.5 * ells_out*(ells_out+1) )) if fftlog_way else ql.spec.cl2cfft(np.nan_to_num(1 / (0.5 * np.arange(self.lmax_out+1)*(np.arange(self.lmax_out+1)+1) )),exp.pix).fft

        # itgnd factors from Limber projection (adapted to hmvec conventions)
        Iyyy_itgnd = 4 * tls.limber_itgrnd_kernel(hcos, 4) * tls.CIB_window(hcos) * tls.y_window(hcos)**3
        IIyy_itgnd = 2 * tls.limber_itgrnd_kernel(hcos, 4) * tls.CIB_window(hcos)**2 * tls.y_window(hcos)**2
        IyIy_itgnd = 2 * IIyy_itgnd
        yIII_itgnd = 4 * tls.limber_itgrnd_kernel(hcos, 4) * tls.CIB_window(hcos)**3 * tls.y_window(hcos)
        kIy_itgnd  = 4 * tls.limber_itgrnd_kernel(hcos, 3) * tls.CIB_window(hcos) * tls.y_window(hcos) \
                     * tls.my_lensing_window(hcos, 1100.)

        # Integrate over z
        exp.biases['mixed']['trispec']['1h'] = np.trapz( Iyyy_itgnd*Iyyy_1h + IIyy_itgnd*IIyy_1h
                                                         + IyIy_itgnd*IyIy_1h + yIII_itgnd*yIII_1h, hcos.zs, axis=-1)
        exp.biases['mixed']['trispec']['2h'] = np.trapz( Iyyy_itgnd*(Iyyy_2h_1_3+Iyyy_2h_2_2)
                                                         + IIyy_itgnd*(IIyy_2h_1_3+IIyy_2h_2_2)
                                                         + IyIy_itgnd*(IyIy_2h_1_3+IyIy_2h_2_2)
                                                         + yIII_itgnd*(yIII_2h_1_3+yIII_2h_2_2), hcos.zs, axis=-1)
        exp.biases['mixed']['prim_bispec']['1h'] = conversion_factor * np.trapz( oneH_cross*kIy_itgnd, hcos.zs, axis=-1)
        exp.biases['mixed']['prim_bispec']['2h'] = conversion_factor * np.trapz( twoH_cross*kIy_itgnd, hcos.zs, axis=-1)

        if get_secondary_bispec_bias:
            # Perm factor of 4 implemented in the get_secondary_bispec_bias_at_L() function
            exp.biases['mixed']['second_bispec']['1h'] = 2 * np.trapz( oneH_second_bispec * kIy_itgnd, hcos.zs, axis=-1)
            exp.biases['mixed']['second_bispec']['2h'] = 2 * np.trapz( twoH_second_bispec * kIy_itgnd, hcos.zs, axis=-1)
            exp.biases['second_bispec_bias_ells'] = L_array_sec_bispec_bias

        if fftlog_way:
            exp.biases['ells'] = ells_out
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['mixed']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return

    def get_mixed_cross_biases(self, exp, gzs, gdndz, fftlog_way=True, bin_width_out=30, survey_name='LSST',
                               damp_1h_prof=True, gal_consistency=False, max_workers=None):
        """
        Calculate the mixed tsz-cib  biases to the cross-correlation of CMB lensing with a galaxy survey, (C^{g\phi}_L)
        given an "experiment" object (defined in qest.py)
        Input:
            * exp = a qest.experiment object
            * gzs = array. Redsfhits at which the dndz is defined. Assumed to be zero otherwise.
            * gdndz = array of same size as gzs. The dndz of the galaxy sample (does not need to be normalized)
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) survey_name = str. Name labelling the HOD characterizing the survey we are x-ing lensing with
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
            * (optional) gal_consistency = Bool. Whether to impose consistency condition on g to correct for missing
                              low mass halos in integrals a la Schmidt 15. Typically not needed
            * (optional) max_workers = int. Max number of parallel workers to launch. Default is # the machine has
        """
        hcos = self.hcos
        if gal_consistency:
            self.get_galaxy_consistency(exp, survey_name)
        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)
        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        exp.tsz_filter = exp.get_tsz_filter()

        # Output ells
        ells_out = np.logspace(np.log10(2), np.log10(self.lmax_out))
        # Get the nodes, weights and matrices needed for Gaussian quadrature of QE integral
        exp.get_weights_mat_total(ells_out)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j;
        twoH_cross = oneH_cross.copy()

        # If using FFTLog, we can compress the normalization to 1D
        if fftlog_way:
            norm_bin_width = 40  # These are somewhat arbitrary
            lmin = 1  # These are somewhat arbitrary
            lbins = np.arange(lmin, exp.lmax, norm_bin_width)
            exp.qe_norm_compressed = np.interp(ells_out, exp.qe_norm.get_ml(lbins).ls, exp.qe_norm.get_ml(lbins).specs['cl'])
        else:
            exp.qe_norm_compressed = exp.qe_norm

        # Run in parallel
        print('Launching parallel processes...')
        hm_minimal = Hm_minimal(self)
        exp_minimal = qest.Exp_minimal(exp)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            n = len(hcos.zs)
            outputs = executor.map(mixed_cross_itgrnds_each_z, np.arange(n), n * [ells_out], n * [fftlog_way],
                                   n * [damp_1h_prof], n * [exp_minimal], n * [hm_minimal], n * [survey_name])

            for idx, itgnds_at_i in enumerate(outputs):
                oneH_cross[...,idx], twoH_cross[...,idx] = itgnds_at_i

        # itgnd factors from Limber projection (adapted to hmvec conventions)
        gIy_itgnd = 2 * tls.scale_sz(exp.freq_GHz) * self.T_CMB * tls.limber_itgrnd_kernel(hcos, 3)\
                    * tls.gal_window(hcos, hcos.zs, gzs, gdndz) * tls.CIB_window(hcos) * tls.y_window(hcos)

        # Integrate over z
        exp.biases['mixed']['cross_w_gals']['1h'] = np.trapz( oneH_cross*gIy_itgnd, hcos.zs, axis=-1)
        exp.biases['mixed']['cross_w_gals']['2h'] = np.trapz(twoH_cross*gIy_itgnd, hcos.zs, axis=-1)

        if fftlog_way:
            exp.biases['ells'] = ells_out
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['mixed']['cross_w_gals']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['cross_w_gals']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['cross_w_gals']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['cross_w_gals']['2h']).get_ml(lbins).specs['cl']
            return

    def get_bias_to_delensed_clbb(self, exp, get_cib=True, get_tsz=True, get_mixed=False, fftlog_way=True,
                                  get_secondary_bispec_bias=False, bin_width_out=30, \
                                  bin_width_out_second_bispec_bias=250, parallelise_secondbispec=True, lmax_clkk=None,
                                  bin_width=30, lmin_clbb=2, lmax_clbb=1000):
        """
        Calculate the leading biases to the power spectrum of delensed B-mode, eqns (24) and (25) in arXiv:2205.09000
        Input:
            * exp = a qest.experiment object
            * (optional) get_cib = Boolean. Whether or not to calculate the contribution from CIB
            * (optional) get_tsz = Boolean. Whether or not to calculate the contribution from tSZ
            * (optional) get_mixed = Boolean. Whether or not to calculate the contribution from CIB-tSZ correlation
            * (optional) fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D qiucklens
            * (optional) get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
            * (optional) bin_width_out = int. Bin width of the output lensing reconstruction
            * (optional) bin_width_out_second_bispec_bias = int. Bin width of the output secondary bispectrum bias
            * (optional) lmax_clkk = int. The size of the array containing the bias to clkk. In practice, anything
                         larger than 2000 is irrelevant for large-scale-B-mode delensing
            * (optional) bin_width = int. Bin width of the output clbb spectra
            * (optional) lmin_clbb = int. lmin of the output clbb spectra
            * (optional) lmax_clbb = int. lmax of the output clbb spectra
        Returns:
            * ells, cl_Btemp_x_Blens_bias, cl_Btemp_x_Btemp_bias, cl_Bdel_x_Bdel_bias
              where cl_Btemp_x_Blens_bias, cl_Btemp_x_Btemp_bias, cl_Bdel_x_Bdel_bias are numpy arrays containing the
              bias to the binned cls describe by the variable name, and ells are the corresponding bin centers
        """
        if lmax_clkk==None:
            lmax_clkk = exp.lmax
        ells_for_clkk = np.arange(lmax_clkk + 1)

        # Initialize array with all biases to kappa rec auto
        clkk_bias_tot = np.zeros(lmax_clkk+1, dtype='complex128')
        # Initialize array with biases to kappa rec cross true kappa (i.e., just the primary bispectrum bias, without
        # the permutation factor of 2
        clkcross_bias_tot = np.zeros(lmax_clkk+1, dtype='complex128')

        # Define bins for the output spectra
        lbins = np.arange(lmin_clbb, lmax_clbb, bin_width)

        which_coupling_list = ['prim_bispec', 'trispec']
        if get_secondary_bispec_bias:
            which_coupling_list.append('second_bispec')

        which_bias_list = []
        if get_cib:
            self.get_cib_auto_biases(exp, get_secondary_bispec_bias=get_secondary_bispec_bias)
            which_bias_list.append('cib')
        if get_tsz:
            self.get_tsz_auto_biases(exp, get_secondary_bispec_bias=get_secondary_bispec_bias)
            which_bias_list.append('tsz')
        if get_mixed:
            self.get_mixed_auto_biases(exp, get_secondary_bispec_bias=get_secondary_bispec_bias)
            which_bias_list.append('mixed')

        for which_bias in which_bias_list:
            for which_coupling in which_coupling_list:
                if which_coupling=='second_bispec':
                    ells = exp.biases['second_bispec_bias_ells']
                else:
                    ells = exp.biases['ells']

                for which_term in ['1h', '2h']:
                    clkk_bias_tot += np.nan_to_num(np.interp(ells_for_clkk, ells,
                                               exp.biases[which_bias][which_coupling][which_term]))
                    if which_coupling=='prim_bispec':
                        # The prim bispec bias to the recXtrueKappa is half of the prim bispec bias to the auto
                        clkcross_bias_tot += np.nan_to_num(0.5 * np.interp(ells_for_clkk, ells,
                                                             exp.biases[which_bias][which_coupling][which_term]))

        # Now use clkk_bias_tot to get the bias to C_l^{B^{template}x\tilde{B}} and C_l^{B^{template}xB^{template}}
        # TODO: speed up calculate_cl_bias() by using fftlog
        cl_Btemp_x_Blens_bias_bcl = tls.calculate_cl_bias(exp.pix,
                                                          exp.W_E(lmax_clkk) * exp.sky.cmb[0, 0].funlensedEE(
                                                              ells_for_clkk),
                                                          exp.W_phi(lmax_clkk) * clkcross_bias_tot, lbins)
        cl_Btemp_x_Btemp_bias_bcl = tls.calculate_cl_bias(exp.pix, exp.W_E(lmax_clkk) ** 2 * exp.clee_tot,
                                                          exp.W_phi(lmax_clkk) ** 2 * clkk_bias_tot, lbins)
        cl_Bdel_x_Bdel_bias_array = - 2 * cl_Btemp_x_Blens_bias_bcl.specs['cl'] + cl_Btemp_x_Btemp_bias_bcl.specs['cl']

        return cl_Btemp_x_Blens_bias_bcl.ls, cl_Btemp_x_Blens_bias_bcl.specs['cl'],\
               cl_Btemp_x_Btemp_bias_bcl.specs['cl'], cl_Bdel_x_Bdel_bias_array

#
# On to the integrands at each z
#

def tsZ_auto_itgrnds_each_z(i, ells_out, fftlog_way, get_secondary_bispec_bias, parallelise_secondbispec,
                            damp_1h_prof, L_array_sec_bispec_bias, exp_minimal, hm_minimal):
    """
    Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
    Input:
        * i = int. Index of the ith redshift in the halo model calculation
        * fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
        * get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
        * parallelise_secondbispec = bool.
        * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        * L_array_sec_bispec_bias = 1D np array. Ls to be used in the secondary bispec bias calculation
        * exp_minimal = instance of qest.exp_minimal(exp)
        * hm_minimal = instance of biases.hm_minimal(hm_framework)
    """
    print(f'Now in parallel loop {i}')
    #nx = hm_minimal.lmax_out + 1 if fftlog_way else exp_minimal.pix.nx
    nx = len(ells_out) if fftlog_way else exp_minimal.pix.nx
    ells_in = np.arange(0,exp_minimal.lmax+1)
    pix_sbbs = ql.maps.cfft(exp_minimal.nx_secbispec, exp_minimal.dx_secbispec)

    # Temporary storage
    # TODO: do these arrays need to be complex?
    itgnd_1h_4pt = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j
    itgnd_1h_cross = itgnd_1h_4pt.copy();
    itgnd_2h_1_3_trispec = itgnd_1h_4pt.copy();
    itgnd_2h_2_2_trispec = itgnd_1h_4pt.copy();
    itgnd_2h_1g = itgnd_1h_4pt.copy();
    itgnd_2h_2g = itgnd_1h_4pt.copy();
    integ_1h_for_2htrispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) # TODO: not sure what to do here with nx for QL
    integ_k_second_bispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses])
    itgnd_1h_second_bispec = np.zeros([len(L_array_sec_bispec_bias), hm_minimal.nMasses]) + 0j
    itgnd_2h_second_bispec = itgnd_1h_second_bispec.copy()
    itgnd_2h_ky_y = itgnd_1h_cross.copy();

    # To keep QE calls tidy, define
    QE = lambda prof_1, prof_2 : qest.get_TT_qe(fftlog_way, ells_out, prof_1, exp_minimal.qe_norm,
                                           exp_minimal.pix, exp_minimal.lmax, exp_minimal.cltt_tot, exp_minimal.ls,
                                           exp_minimal.cl_len.cltt, exp_minimal.qest_lib, exp_minimal.ivf_lib, prof_2,
                                           weights_mat_total=exp_minimal.weights_mat_total, nodes=exp_minimal.nodes)

    # Project the matter power spectrum for two-halo terms
    pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i])(ells_in)
    pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i])(ells_out)
    if not fftlog_way:
        pk_of_l = ql.spec.cl2cfft(pk_of_l, exp_minimal.pix).fft
        pk_of_L = ql.spec.cl2cfft(pk_of_L, exp_minimal.pix).fft

    # Integral over M for 2halo trispectrum. This will later go into a QE
    for j, m in enumerate(hm_minimal.ms):
        kap = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j])(ells_in)
        integ_k_second_bispec[..., j] = kap * hm_minimal.ms_rescaled[j] * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

        if m < exp_minimal.massCut:
            y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                          hm_minimal.pk_profiles['y'][i, j])(ells_in)
            integ_1h_for_2htrispec[..., j] = y * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

    # Integrals over single profile in 2h (for 1-3 trispec and 2h bispec biases)
    int_over_M_of_profile = pk_of_l * (np.trapz(integ_1h_for_2htrispec, hm_minimal.ms, axis=-1) + hm_minimal.y_consistency[i])
    kap_int = pk_of_l * (np.trapz(integ_k_second_bispec, hm_minimal.ms, axis=-1) + hm_minimal.m_consistency[i])

    # M integral
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.pk_profiles['y'][i, j])(ells_in)
        kap = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j])(ells_out)

        kfft = kap * hm_minimal.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap, exp_minimal.pix).fft * hm_minimal.ms_rescaled[j]

        phicfft = QE(y, y)
        phicfft_mixed = QE(y, int_over_M_of_profile)
        # Consider damping the profiles at low k in 1h terms to avoid it exceeding many-halo amplitude
        if damp_1h_prof:
            y_damp = exp_minimal.tsz_filter * \
                     tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.pk_profiles['y'][i, j]
                                  * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_in)
            kap_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                                    hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j]
                                    * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_out)
            kfft_damp = kap_damp * hm_minimal.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp, exp_minimal.pix).fft * \
                                                                          hm_minimal.ms_rescaled[j]
            phicfft_damp = QE(y_damp, y_damp)
        else:
            y_damp = y; kfft_damp = kfft; phicfft_damp = phicfft

        # Accumulate the itgnds
        itgnd_1h_cross[..., j] = phicfft_damp * np.conjugate(kfft_damp) * hm_minimal.nzm[i, j]
        itgnd_1h_4pt[..., j] = phicfft_damp * np.conjugate(phicfft_damp) * hm_minimal.nzm[i, j]
        itgnd_2h_1g[..., j] = np.conjugate(kfft) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_2g[..., j] = phicfft * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_1_3_trispec[..., j] = phicfft * np.conjugate(phicfft_mixed) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_2_2_trispec[..., j] = phicfft * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_ky_y[..., j] = np.conjugate(kfft) * phicfft_mixed\
                                * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

        if get_secondary_bispec_bias:
            # To keep QE calls tidy, define
            sec_bispec_rec = lambda prof_1, k_prof, prof_2: sbbs.get_sec_bispec_bias(L_array_sec_bispec_bias,
                                                                                    exp_minimal.qe_norm_at_lbins_sec_bispec,
                                                                                    pix_sbbs, exp_minimal.cl_len, exp_minimal.cl_unl,
                                                                                    exp_minimal.cltt_tot, prof_1, k_prof,
                                                                                    projected_fg_profile_2=prof_2,
                                                                                    parallelise=parallelise_secondbispec)

            # Get the kappa map, up to lmax rather than lmax_out as was needed in other terms
            kap_secbispec = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                         hm_minimal.uk_profiles['nfw'][i, j])(ells_in)
            if damp_1h_prof:
                kap_secbispec_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                             hm_minimal.uk_profiles['nfw'][i, j]
                                             * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))))(ells_in)
            else:
                kap_secbispec_damp = kap_secbispec

            sec_bispec_rec_1h = sec_bispec_rec(y_damp, kap_secbispec_damp * hm_minimal.ms_rescaled[j], y_damp)
            sec_bispec_rec_2h = sec_bispec_rec(y, kap_int, y) \
                                + 2*sec_bispec_rec(int_over_M_of_profile, kap_secbispec_damp * hm_minimal.ms_rescaled[j], y)

            itgnd_1h_second_bispec[..., j] = hm_minimal.nzm[i, j] * sec_bispec_rec_1h
            itgnd_2h_second_bispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * sec_bispec_rec_2h

    # Perform the m integrals
    oneH_4pt_at_i = np.trapz(itgnd_1h_4pt, hm_minimal.ms, axis=-1)
    oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)
    oneH_second_bispec_at_i = np.trapz(itgnd_1h_second_bispec, hm_minimal.ms, axis=-1)
    twoH_second_bispec_at_i = np.trapz(itgnd_2h_second_bispec, hm_minimal.ms, axis=-1)

    # TODO: apply consistency to 2-2 trispectrum
    twoH_2_2_at_i = 2 * np.trapz(itgnd_2h_2_2_trispec, hm_minimal.ms, axis=-1) ** 2 * pk_of_L
    twoH_1_3_at_i = 4 * np.trapz(itgnd_2h_1_3_trispec, hm_minimal.ms, axis=-1)
    tmpCorr = np.trapz(itgnd_2h_1g, hm_minimal.ms, axis=-1)
    twoH_cross_at_i = np.trapz(itgnd_2h_2g, hm_minimal.ms, axis=-1) * (tmpCorr + hm_minimal.m_consistency[i]) * pk_of_L \
                      + np.trapz(2*itgnd_2h_ky_y, hm_minimal.ms, axis=-1)
    return oneH_4pt_at_i, oneH_cross_at_i, twoH_2_2_at_i, twoH_1_3_at_i, twoH_cross_at_i, oneH_second_bispec_at_i, \
           twoH_second_bispec_at_i

def tsZ_cross_itgrnds_each_z(i, ells_out, fftlog_way, damp_1h_prof, exp_minimal, hm_minimal, survey_name):
    """
    Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
    Input:
        * i = int. Index of the ith redshift in the halo model calculation
        * fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
        * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        * exp_minimal = instance of qest.exp_minimal(exp)
        * hm_minimal = instance of biases.hm_minimal(hm_framework)
    """
    print(f'Now in parallel loop {i}')
    nx = hm_minimal.lmax_out + 1 if fftlog_way else exp_minimal.pix.nx
    # Temporary storage
    itgnd_1h_cross = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j
    itgnd_2h_2g = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j
    itgnd_2h_1g = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j
    itgnd_2h_ky_y = itgnd_1h_cross.copy();
    itgnd_y_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) # TODO: not sure what to do here with nx for QL

    # To keep QE calls tidy, define
    QE = lambda prof_1, prof_2 : qest.get_TT_qe(fftlog_way, ells_out, prof_1, exp_minimal.qe_norm,
                                           exp_minimal.pix, exp_minimal.lmax, exp_minimal.cltt_tot, exp_minimal.ls,
                                           exp_minimal.cl_len.cltt, exp_minimal.qest_lib, exp_minimal.ivf_lib, prof_2,
                                           weights_mat_total=exp_minimal.weights_mat_total, nodes=exp_minimal.nodes)

    # Project the matter power spectrum for two-halo terms
    pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=exp_minimal.lmax)
    pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=hm_minimal.lmax_out)
    if not fftlog_way:
        pk_of_l = ql.spec.cl2cfft(pk_of_l, exp_minimal.pix).fft
        pk_of_L = ql.spec.cl2cfft(pk_of_L, exp_minimal.pix).fft

    # Integral over M for 2halo trispectrum. This will later go into a QE
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                      hm_minimal.pk_profiles['y'][i, j], ellmax=exp_minimal.lmax)
        itgnd_y_for_2hbispec[..., j] = y * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

    int_over_M_of_y = pk_of_l * (np.trapz(itgnd_y_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.y_consistency[i])

    # M integral.
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                      hm_minimal.pk_profiles['y'][i, j], ellmax=exp_minimal.lmax)
        # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
        # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 #TODO: why do you say that?
        gal = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j], ellmax=hm_minimal.lmax_out)
        # TODO: should ngal in denominator depend on z? ms_rescaled doesn't
        galfft = gal / hm_minimal.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal, exp_minimal.pix).fft / \
                                                                            hm_minimal.hods[survey_name]['ngal'][i]
        phicfft = QE(y, y)
        phicfft_y_intofy = QE(y, int_over_M_of_y)

        # Consider damping the profiles at low k in 1h terms to avoid it exceeding many-halo amplitude
        if damp_1h_prof:
            y_damp = exp_minimal.tsz_filter \
                     * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                    hm_minimal.pk_profiles['y'][i, j]
                                    * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=exp_minimal.lmax)
            gal_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                    hm_minimal.uk_profiles['nfw'][i, j]
                                    * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=hm_minimal.lmax_out)
            galfft_damp = gal_damp / hm_minimal.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal_damp,
                                                                                                          exp_minimal.pix).fft / \
                                                                                          hm_minimal.hods[survey_name][
                                                                                              'ngal'][i]
            phicfft_damp = QE(y_damp, y_damp)
        else:
            y_damp = y; gal_damp = gal; galfft_damp = galfft; phicfft_damp = phicfft

        # Accumulate the itgnds
        mean_Ngal = hm_minimal.hods[survey_name]['Nc'][i, j] + hm_minimal.hods[survey_name]['Ns'][i, j]
        itgnd_1h_cross[..., j] = mean_Ngal * phicfft_damp * np.conjugate(galfft_damp) * hm_minimal.nzm[i, j]
        itgnd_2h_1g[..., j] = mean_Ngal * np.conjugate(galfft) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_2g[..., j] = phicfft * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_ky_y[..., j] = mean_Ngal * np.conjugate(galfft) * phicfft_y_intofy\
                                * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

    # Perform the m integrals
    oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)

    tmpCorr = np.trapz(itgnd_2h_1g, hm_minimal.ms, axis=-1)
    twoH_cross_at_i = np.trapz(itgnd_2h_2g, hm_minimal.ms, axis=-1) * (tmpCorr + hm_minimal.g_consistency[i]) * pk_of_L \
                      + np.trapz(2*itgnd_2h_ky_y, hm_minimal.ms, axis=-1)
    return oneH_cross_at_i, twoH_cross_at_i

def tsZ_ps_itgrnds_each_z(i, ells_out, damp_1h_prof, exp_minimal, hm_minimal):
    """
    Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
    Input:
        * i = int. Index of the ith redshift in the halo model calculation
        * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        * exp_minimal = instance of qest.exp_minimal(exp)
        * hm_minimal = instance of biases.hm_minimal(hm_framework)
    """
    nx = hm_minimal.lmax_out + 1
    # Temporary storage
    itgnd_1h_ps_tSZ = np.zeros([nx, hm_minimal.nMasses]) + 0j

    # M integral.
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        # project the galaxy profiles
        if damp_1h_prof:
            y = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                             hm_minimal.ks,
                             hm_minimal.pk_profiles['y'][i, j] * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))),
                             ellmax=hm_minimal.lmax_out)
        else:
            y = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.pk_profiles['y'][i, j],
                             ellmax=hm_minimal.lmax_out)
        # Accumulate the itgnds
        itgnd_1h_ps_tSZ[:, j] = y * np.conjugate(y) * hm_minimal.nzm[i, j]

        # Perform the m integrals
    oneH_ps_tz_at_i = np.trapz(itgnd_1h_ps_tSZ, hm_minimal.ms, axis=-1)
    return oneH_ps_tz_at_i


def cib_auto_itgrnds_each_z(i, ells_out, fftlog_way, get_secondary_bispec_bias, parallelise_secondbispec,
                            damp_1h_prof, L_array_sec_bispec_bias, exp_minimal, hm_minimal):
    """
    Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
    Input:
        * i = int. Index of the ith redshift in the halo model calculation
        * fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
        * get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
        * parallelise_secondbispec = bool.
        * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        * L_array_sec_bispec_bias = 1D np array. Ls to be used in the secondary bispec bias calculation
        * exp_minimal = instance of qest.exp_minimal(exp)
        * hm_minimal = instance of biases.hm_minimal(hm_framework)
    """
    print(f'Now in parallel loop {i}')
    nx = hm_minimal.lmax_out + 1 if fftlog_way else exp_minimal.pix.nx
    pix_sbbs = ql.maps.cfft(exp_minimal.nx_secbispec, exp_minimal.dx_secbispec)

    # Temporary storage
    itgnd_1h_cross = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j
    itgnd_1h_IIII = itgnd_1h_cross.copy();
    itgnd_2h_k = itgnd_1h_cross.copy()
    itgnd_2h_II = itgnd_1h_cross.copy();
    itgnd_2h_IintIII = itgnd_1h_cross.copy()
    itgnd_2h_kI_I= itgnd_1h_cross.copy();
    integ_1h_for_2htrispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) if fftlog_way else np.zeros( [nx, nx, hm_minimal.nMasses])
    integ_k_second_bispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses])
    itgnd_1h_second_bispec = np.zeros([len(L_array_sec_bispec_bias), hm_minimal.nMasses]) + 0j
    itgnd_2h_second_bispec = itgnd_1h_second_bispec.copy()

    # To keep QE calls tidy, define
    QE = lambda prof_1, prof_2 : qest.get_TT_qe(fftlog_way, ells_out, prof_1, exp_minimal.qe_norm,
                                           exp_minimal.pix, exp_minimal.lmax, exp_minimal.cltt_tot, exp_minimal.ls,
                                           exp_minimal.cl_len.cltt, exp_minimal.qest_lib, exp_minimal.ivf_lib, prof_2,
                                           weights_mat_total=exp_minimal.weights_mat_total, nodes=exp_minimal.nodes)

    # Project the matter power spectrum for two-halo terms
    pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=exp_minimal.lmax)
    pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=hm_minimal.lmax_out)
    if not fftlog_way:
        pk_of_l = ql.spec.cl2cfft(pk_of_l, exp_minimal.pix).fft
        pk_of_L = ql.spec.cl2cfft(pk_of_L, exp_minimal.pix).fft

    # Integral over M for 2halo trispectrum. This will later go into a QE
    for j, m in enumerate(hm_minimal.ms):
        kap = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        integ_k_second_bispec[..., j] = kap * hm_minimal.ms_rescaled[j] * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

        if m < exp_minimal.massCut:
            # project the galaxy profiles
            u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                             hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
            u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
            u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

            integ_1h_for_2htrispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (u_cen + u_sat)

    # Integrals over single profile in 2h (for 1-3 trispec and 2h bispec biases)
    Iint = pk_of_l * (np.trapz(integ_1h_for_2htrispec, hm_minimal.ms, axis=-1) + hm_minimal.I_consistency[i])
    kap_int = pk_of_l * (np.trapz(integ_k_second_bispec, hm_minimal.ms, axis=-1) + hm_minimal.m_consistency[i])

    # M integral.
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        # project the galaxy profiles
        u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                         hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
        u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

        phicfft_ucen_usat = QE(u_cen, u_sat)
        phicfft_usat_usat = QE(u_sat, u_sat)
        phicfft_Iint_ucen = QE(Iint, u_cen)
        phicfft_Iint_usat = QE(Iint, u_sat)

        # Get the kappa map
        kap = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j], ellmax=hm_minimal.lmax_out)
        kfft = kap * hm_minimal.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap, exp_minimal.pix).fft * hm_minimal.ms_rescaled[j]

        if damp_1h_prof:
            # Damp the profiles on large scales when calculating 1h terms
            # Note that we are damping u_sat, but leaving u_cen as is, because it's always at the center
            u_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                  hm_minimal.uk_profiles['nfw'][i, j] * (
                                              1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))),
                                  ellmax=exp_minimal.lmax)
            # TODO: check that the correct way to damp u_cen is to leave it as is
            u_sat_damp = hm_minimal.CIB_satellite_filter[:, i, j] * u_damp

            phicfft_ucen_usat_damp = QE(u_cen, u_sat_damp)
            phicfft_usat_usat_damp = QE(u_sat_damp, u_sat_damp)

            # Get the kappa map
            kap_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                                    hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j]
                                    * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=hm_minimal.lmax_out)
            kfft_damp = kap_damp * hm_minimal.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp, exp_minimal.pix).fft * \
                                                                          hm_minimal.ms_rescaled[j]
        else:
            u_sat_damp = u_sat;
            phicfft_ucen_usat_damp = phicfft_ucen_usat;
            phicfft_usat_usat_damp = phicfft_usat_usat;
            kfft_damp = kfft

        # Accumulate the itgnds
        itgnd_1h_cross[..., j] = hm_minimal.nzm[i, j] * np.conjugate(kfft_damp) * (phicfft_usat_usat_damp +
                                                                             2 * phicfft_ucen_usat_damp)
        itgnd_1h_IIII[..., j] = hm_minimal.nzm[i, j] * (phicfft_usat_usat_damp * np.conjugate(phicfft_usat_usat_damp)
                                                  + 4 * phicfft_ucen_usat_damp * np.conjugate(
                    phicfft_usat_usat_damp))

        itgnd_2h_k[..., j] = np.conjugate(kfft) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_II[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (phicfft_usat_usat + 2 * phicfft_ucen_usat)
        itgnd_2h_IintIII[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] \
                                   * (phicfft_Iint_ucen * np.conjugate(phicfft_usat_usat)
                                      + phicfft_Iint_usat * np.conjugate(2 * phicfft_ucen_usat + phicfft_usat_usat))
        itgnd_2h_II[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (phicfft_usat_usat + 2 * phicfft_ucen_usat)
        itgnd_2h_kI_I[..., j] = np.conjugate(kfft) * (phicfft_Iint_ucen + phicfft_Iint_usat)\
                                * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

        if get_secondary_bispec_bias:
            # To keep QE calls tidy, define
            sec_bispec_rec = lambda prof_1, k_prof, prof_2: sbbs.get_sec_bispec_bias(L_array_sec_bispec_bias,
                                                                                    exp_minimal.qe_norm_at_lbins_sec_bispec,
                                                                                    pix_sbbs, exp_minimal.cl_len,
                                                                                    exp_minimal.cl_unl,
                                                                                    exp_minimal.cltt_tot, prof_1,
                                                                                    k_prof,
                                                                                    projected_fg_profile_2=prof_2,
                                                                                    parallelise=parallelise_secondbispec)
            # Get the kappa map, up to lmax rather than lmax_out as was needed in other terms
            kap_secbispec = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                         hm_minimal.uk_profiles['nfw'][i, j],
                                         ellmax=exp_minimal.lmax)
            if damp_1h_prof:
                kap_secbispec_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                             hm_minimal.uk_profiles['nfw'][i, j]
                                             * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))),
                                             ellmax=exp_minimal.lmax)
            else:
                kap_secbispec_damp = kap_secbispec
            sec_bispec_rec_1h = sec_bispec_rec(2*u_cen + u_sat_damp, kap_secbispec_damp * hm_minimal.ms_rescaled[j],
                                               u_sat_damp)
            sec_bispec_rec_2h = sec_bispec_rec(2*u_cen + u_sat, kap_int, u_sat) \
                                + 2*sec_bispec_rec(u_cen + u_sat, kap_secbispec_damp * hm_minimal.ms_rescaled[j], Iint)

            itgnd_1h_second_bispec[..., j] = hm_minimal.nzm[i, j] * sec_bispec_rec_1h
            itgnd_2h_second_bispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * sec_bispec_rec_2h


    # Perform the m integrals
    IIII_1h_at_i = np.trapz(itgnd_1h_IIII, hm_minimal.ms, axis=-1)

    oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)
    oneH_second_bispec_at_i = np.trapz(itgnd_1h_second_bispec, hm_minimal.ms, axis=-1)
    twoH_second_bispec_at_i = np.trapz(itgnd_2h_second_bispec, hm_minimal.ms, axis=-1)

    IIII_2h_1_3_at_i = 4 * np.trapz(itgnd_2h_IintIII, hm_minimal.ms, axis=-1)
    # TODO: implement consistency for 2-2 trispectrum
    IIII_2h_2_2_at_i = 2 * np.trapz(itgnd_2h_II, hm_minimal.ms, axis=-1) ** 2 * pk_of_L

    tmpCorr = np.trapz(itgnd_2h_k, hm_minimal.ms, axis=-1)
    twoH_cross_at_i = np.trapz(itgnd_2h_II, hm_minimal.ms, axis=-1) \
                         * (tmpCorr + hm_minimal.m_consistency[i]) * pk_of_L \
                      + np.trapz(2*itgnd_2h_kI_I, hm_minimal.ms, axis=-1)
    # TODO: save memory by return 1-3 and 2-2 trispectra together. This will also be more consistent w what you do for primary bispectra
    return IIII_1h_at_i, oneH_cross_at_i, IIII_2h_2_2_at_i, IIII_2h_1_3_at_i, twoH_cross_at_i, oneH_second_bispec_at_i,\
           twoH_second_bispec_at_i

def cib_cross_itgrnds_each_z(i, ells_out, fftlog_way, damp_1h_prof, exp_minimal, hm_minimal, survey_name):
    """
    Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
    Input:
        * i = int. Index of the ith redshift in the halo model calculation
        * fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
        * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        * exp_minimal = instance of qest.exp_minimal(exp)
        * hm_minimal = instance of biases.hm_minimal(hm_framework)
    """
    print(f'Now in parallel loop {i}')
    nx = hm_minimal.lmax_out + 1 if fftlog_way else exp_minimal.pix.nx
    # Temporary storage
    itgnd_1h_cross = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j;
    itgnd_2h_k = itgnd_1h_cross.copy();
    itgnd_2h_II = itgnd_1h_cross.copy()
    itgnd_2h_kI_I= itgnd_1h_cross.copy();
    itgnd_I_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) # TODO: not sure what to do here with nx for QL

    # To keep QE calls tidy, define
    QE = lambda prof_1, prof_2 : qest.get_TT_qe(fftlog_way, ells_out, prof_1, exp_minimal.qe_norm,
                                           exp_minimal.pix, exp_minimal.lmax, exp_minimal.cltt_tot, exp_minimal.ls,
                                           exp_minimal.cl_len.cltt, exp_minimal.qest_lib, exp_minimal.ivf_lib, prof_2,
                                           weights_mat_total=exp_minimal.weights_mat_total, nodes=exp_minimal.nodes)

    # Project the matter power spectrum for two-halo terms
    pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=exp_minimal.lmax)
    pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=hm_minimal.lmax_out)
    if not fftlog_way:
        pk_of_l = ql.spec.cl2cfft(pk_of_l, exp_minimal.pix).fft
        pk_of_L = ql.spec.cl2cfft(pk_of_L, exp_minimal.pix).fft

    # Integral over M for 2halo trispectrum. This will later go into a QE
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        # project the galaxy profiles
        u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                         hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
        u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

        itgnd_I_for_2hbispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (u_cen + u_sat)

    int_over_M_of_I = pk_of_l * (np.trapz(itgnd_I_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.I_consistency[i])

    # M integral.
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        # project the galaxy profiles
        u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                         hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
        u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

        # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
        # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 # TODO:Why?
        gal = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j], ellmax=hm_minimal.lmax_out)
        galfft = gal / hm_minimal.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal, exp_minimal.pix).fft / \
                                                                            hm_minimal.hods[survey_name]['ngal'][i]

        phicfft_ucen_usat = QE(u_cen, u_sat)
        phicfft_usat_usat = QE(u_sat, u_sat)
        phicfft_I_intofI = QE(u_cen + u_sat, int_over_M_of_I)

        if damp_1h_prof:
            u_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                  hm_minimal.uk_profiles['nfw'][i, j] * (
                                              1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))),
                                  ellmax=exp_minimal.lmax)
            u_sat_damp = hm_minimal.CIB_satellite_filter[:, i, j] * u_damp
            gal_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                                    hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j]
                                    * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=hm_minimal.lmax_out)
            galfft_damp = gal_damp / hm_minimal.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal_damp,
                                                                                                          exp_minimal.pix).fft / \
                                                                                          hm_minimal.hods[survey_name][
                                                                                              'ngal'][i]

            phicfft_ucen_usat_damp = QE(u_cen, u_sat_damp)
            phicfft_usat_usat_damp = QE(u_sat_damp, u_sat_damp)
        else:
            galfft_damp = galfft;
            phicfft_ucen_usat_damp = phicfft_ucen_usat;
            phicfft_usat_usat_damp = phicfft_usat_usat

        # Accumulate the itgnds
        mean_Ngal = hm_minimal.hods[survey_name]['Nc'][i, j] + hm_minimal.hods[survey_name]['Ns'][i, j]
        itgnd_1h_cross[..., j] = mean_Ngal * np.conjugate(galfft_damp) * hm_minimal.nzm[i, j] * \
                                 (phicfft_usat_usat_damp + 2 * phicfft_ucen_usat_damp)
        itgnd_2h_k[..., j] = mean_Ngal * np.conjugate(galfft) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_II[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] \
                              * ((phicfft_usat_usat + 2 * phicfft_ucen_usat))
        itgnd_2h_kI_I[..., j] = mean_Ngal * np.conjugate(galfft) * phicfft_I_intofI\
                                * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

    oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)

    tmpCorr = np.trapz(itgnd_2h_k, hm_minimal.ms, axis=-1)
    twoH_cross_at_i = np.trapz(itgnd_2h_II, hm_minimal.ms, axis=-1) * (tmpCorr + hm_minimal.g_consistency[i]) * pk_of_L\
                      + np.trapz(2*itgnd_2h_kI_I, hm_minimal.ms, axis=-1)
    return oneH_cross_at_i, twoH_cross_at_i

def mixed_auto_itgrnds_each_z(i, ells_out, fftlog_way, get_secondary_bispec_bias, parallelise_secondbispec,
                            damp_1h_prof, L_array_sec_bispec_bias, exp_minimal, hm_minimal):
    """
    Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
    Input:
        * i = int. Index of the ith redshift in the halo model calculation
        * fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
        * get_secondary_bispec_bias = False. Compute and return the secondary bispectrum bias (slow)
        * parallelise_secondbispec = bool.
        * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        * L_array_sec_bispec_bias = 1D np array. Ls to be used in the secondary bispec bias calculation
        * exp_minimal = instance of qest.exp_minimal(exp)
        * hm_minimal = instance of biases.hm_minimal(hm_framework)
    """
    print(f'Now in parallel loop {i}')
    nx = hm_minimal.lmax_out + 1 if fftlog_way else exp_minimal.pix.nx
    pix_sbbs = ql.maps.cfft(exp_minimal.nx_secbispec, exp_minimal.dx_secbispec)

    # Temporary storage
    itgnd_1h_cross = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j
    itgnd_1h_Iyyy = itgnd_1h_cross.copy(); itgnd_1h_IIyy = itgnd_1h_cross.copy(); itgnd_1h_yIII = itgnd_1h_cross.copy();
    itgnd_2h_k = itgnd_1h_cross.copy(); itgnd_2h_Iy = itgnd_1h_cross.copy(); itgnd_1h_IyIy = itgnd_1h_cross.copy()
    integ_1h_I_for_2htrispec=np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) if fftlog_way else np.zeros([nx,nx, hm_minimal.nMasses])
    integ_1h_y_for_2htrispec = integ_1h_I_for_2htrispec.copy(); itgnd_2h_Iyyy = itgnd_1h_cross.copy();
    itgnd_2h_IIyy = itgnd_1h_cross.copy(); itgnd_2h_IyIy = itgnd_1h_cross.copy();
    itgnd_2h_yIII = itgnd_1h_cross.copy(); itgnd_2h_Iy = itgnd_1h_cross.copy();
    itgnd_2h_yy = itgnd_1h_cross.copy(); itgnd_2h_II = itgnd_1h_cross.copy();
    itgnd_1h_second_bispec = np.zeros([len(L_array_sec_bispec_bias), hm_minimal.nMasses]) + 0j
    itgnd_2h_second_bispec = itgnd_1h_second_bispec.copy()
    itgnd_2h_kinHaloWfg = itgnd_1h_cross.copy();
    integ_k_second_bispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses])

    # To keep QE calls tidy, define
    QE = lambda prof_1, prof_2 : qest.get_TT_qe(fftlog_way, ells_out, prof_1, exp_minimal.qe_norm,
                                           exp_minimal.pix, exp_minimal.lmax, exp_minimal.cltt_tot, exp_minimal.ls,
                                           exp_minimal.cl_len.cltt, exp_minimal.qest_lib, exp_minimal.ivf_lib, prof_2,
                                           weights_mat_total=exp_minimal.weights_mat_total, nodes=exp_minimal.nodes)

    # Project the matter power spectrum for two-halo terms
    pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=exp_minimal.lmax)
    pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=hm_minimal.lmax_out)
    if not fftlog_way:
        pk_of_l = ql.spec.cl2cfft(pk_of_l, exp_minimal.pix).fft
        pk_of_L = ql.spec.cl2cfft(pk_of_L, exp_minimal.pix).fft

    # Integral over M for 2halo trispectrum. This will later go into a QE
    for j, m in enumerate(hm_minimal.ms):
        kap = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        integ_k_second_bispec[..., j] = kap * hm_minimal.ms_rescaled[j] * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

        if m < exp_minimal.massCut:
            # project the galaxy profiles
            u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                             hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
            y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                          hm_minimal.pk_profiles['y'][i, j], ellmax=exp_minimal.lmax)
            u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
            u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

            integ_1h_I_for_2htrispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (u_cen + u_sat)
            integ_1h_y_for_2htrispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * y

    # Integrals over single profile in 2h (for 1-3 trispec and 2h bispec biases)
    Iint = pk_of_l * (np.trapz(integ_1h_I_for_2htrispec, hm_minimal.ms, axis=-1) + hm_minimal.I_consistency[i])
    yint = pk_of_l * (np.trapz(integ_1h_y_for_2htrispec, hm_minimal.ms, axis=-1) + hm_minimal.y_consistency[i])
    kap_int = pk_of_l * (np.trapz(integ_k_second_bispec, hm_minimal.ms, axis=-1) + hm_minimal.m_consistency[i])

    # M integral.
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        # project the galaxy profiles
        y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                      hm_minimal.pk_profiles['y'][i, j], ellmax=exp_minimal.lmax)
        u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                         hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
        u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

        phicfft_ucen_usat = QE(u_cen, u_sat)
        phicfft_usat_usat = QE(u_sat, u_sat)
        phicfft_ucen_y = QE(u_cen, y)
        phicfft_usat_y = QE(u_sat, y)
        phicfft_yy = QE(y, y)

        phicfft_Iint_usat = QE(Iint, u_sat)
        phicfft_Iint_ucen = QE(Iint, u_cen)
        phicfft_Iint_y = QE(Iint, y)
        phicfft_yint_usat = QE(yint, u_sat)
        phicfft_yint_ucen = QE(yint, u_cen)
        phicfft_yint_y = QE(yint, y)

        # Get the kappa map
        kap = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                           hm_minimal.uk_profiles['nfw'][i, j], ellmax=hm_minimal.lmax_out)
        kfft = kap * hm_minimal.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap, exp_minimal.pix).fft * hm_minimal.ms_rescaled[j]

        if damp_1h_prof:
            y_damp = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                               hm_minimal.pk_profiles['y'][i, j]
                                               * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))),
                                               ellmax=exp_minimal.lmax)
            u_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                  hm_minimal.uk_profiles['nfw'][i, j]
                                  * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=exp_minimal.lmax)
            u_sat_damp = hm_minimal.CIB_satellite_filter[:, i, j] * u_damp

            phicfft_ucen_usat_damp = QE(u_cen, u_sat_damp)
            phicfft_usat_usat_damp = QE(u_sat_damp, u_sat_damp)
            phicfft_ucen_y_damp = QE(u_cen, y_damp)
            phicfft_usat_y_damp = QE(u_sat_damp, y_damp)
            phicfft_yy_damp = QE(y_damp, y_damp)

            kap_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                    hm_minimal.uk_profiles['nfw'][i, j]
                                    * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=hm_minimal.lmax_out)
            kfft_damp = kap_damp * hm_minimal.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp, exp_minimal.pix).fft * \
                                                                          hm_minimal.ms_rescaled[j]
        else:
            kfft_damp = kfft;
            phicfft_yy_damp = phicfft_yy;
            phicfft_usat_y_damp = phicfft_usat_y;
            phicfft_ucen_y_damp = phicfft_ucen_y;
            phicfft_usat_usat_damp = phicfft_usat_usat;
            phicfft_ucen_usat_damp = phicfft_ucen_usat;
            u_sat_damp = u_sat;
            y_damp = y

        # Accumulate the itgnds
        itgnd_1h_cross[..., j] = hm_minimal.nzm[i, j] \
                                 * (phicfft_ucen_y_damp + phicfft_usat_y_damp) * np.conjugate(kfft_damp)
        itgnd_1h_Iyyy[..., j] = hm_minimal.nzm[i, j] \
                                * (phicfft_ucen_y_damp + phicfft_usat_y_damp) * np.conjugate(phicfft_yy_damp)
        itgnd_1h_IIyy[..., j] = hm_minimal.nzm[i, j] \
                                * np.conjugate(phicfft_yy_damp) * (phicfft_usat_usat_damp
                                                                   + 2 * phicfft_ucen_usat_damp)
        itgnd_1h_IyIy[..., j] = hm_minimal.nzm[i, j] \
                                * phicfft_usat_y_damp * (2 * phicfft_ucen_y_damp + phicfft_usat_y_damp)
        itgnd_1h_yIII[..., j] = hm_minimal.nzm[i, j] \
                                * (phicfft_ucen_y_damp * phicfft_usat_usat_damp
                                   + phicfft_usat_y_damp * (2 * phicfft_ucen_usat_damp + phicfft_usat_usat_damp))

        itgnd_2h_k[..., j] = np.conjugate(kfft) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_Iy[..., j] = (phicfft_ucen_y + phicfft_usat_y) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_yy[..., j] = phicfft_yy * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_II[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (phicfft_usat_usat + 2 * phicfft_ucen_usat)

        itgnd_2h_Iyyy[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (phicfft_Iint_y * phicfft_yy
                                                                      + 3 * phicfft_yint_y * (phicfft_ucen_y
                                                                                              + phicfft_usat_y))
        itgnd_2h_IIyy[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (2 * phicfft_yy * (phicfft_Iint_ucen
                                                                                        + phicfft_Iint_usat)
                                                                      + 2 * phicfft_yint_y * (2 * phicfft_ucen_usat
                                                                                              + phicfft_usat_usat))
        itgnd_2h_IyIy[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (2 * phicfft_Iint_y * (phicfft_ucen_y
                                                                                            + phicfft_usat_y)
                                                                      + 2 * phicfft_usat_y * (2 * phicfft_Iint_ucen
                                                                                              + phicfft_Iint_usat))
        itgnd_2h_yIII[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * (phicfft_yint_ucen * phicfft_usat_usat
                                                                      + (phicfft_yint_usat + phicfft_Iint_y)
                                                                      * (2 * phicfft_ucen_usat + phicfft_usat_usat)
                                                                      + 2 * phicfft_Iint_ucen * phicfft_usat_y
                                                                      + 2 * phicfft_Iint_usat * (phicfft_ucen_y
                                                                                                 + phicfft_usat_y))
        itgnd_2h_kinHaloWfg[..., j] = np.conjugate(kfft) * ( phicfft_yint_usat + phicfft_yint_usat + phicfft_Iint_y)\
                                * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

        if get_secondary_bispec_bias:
            # To keep QE calls tidy, define
            sec_bispec_rec = lambda prof_1, k_prof, prof_2: sbbs.get_sec_bispec_bias(L_array_sec_bispec_bias,
                                                                                     exp_minimal.qe_norm_at_lbins_sec_bispec,
                                                                                     pix_sbbs, exp_minimal.cl_len,
                                                                                     exp_minimal.cl_unl,
                                                                                     exp_minimal.cltt_tot, prof_1,
                                                                                     k_prof,
                                                                                     projected_fg_profile_2=prof_2,
                                                                                     parallelise=parallelise_secondbispec)

            # Get the kappa map, up to lmax rather than lmax_out as was needed in other terms
            kap_secbispec = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                         hm_minimal.uk_profiles['nfw'][i, j],
                                         ellmax=exp_minimal.lmax)
            if damp_1h_prof:
                kap_secbispec_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                                  hm_minimal.uk_profiles['nfw'][i, j]
                                                  * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))),
                                                  ellmax=exp_minimal.lmax)
            else:
                kap_secbispec_damp = kap_secbispec
            sec_bispec_rec_1h = sec_bispec_rec(u_cen+u_sat_damp, kap_secbispec_damp * hm_minimal.ms_rescaled[j], y_damp)
            sec_bispec_rec_2h = sec_bispec_rec(u_cen+u_sat_damp, kap_int, y) \
                                + sec_bispec_rec(Iint, kap_secbispec_damp * hm_minimal.ms_rescaled[j], y) \
                                + sec_bispec_rec(yint, kap_secbispec_damp * hm_minimal.ms_rescaled[j], u_cen+u_sat)

            itgnd_1h_second_bispec[..., j] = hm_minimal.nzm[i, j] * sec_bispec_rec_1h
            itgnd_2h_second_bispec[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] * sec_bispec_rec_2h

    # Perform the m integrals
    Iyyy_1h_at_i = np.trapz(itgnd_1h_Iyyy, hm_minimal.ms, axis=-1)
    IIyy_1h_at_i = np.trapz(itgnd_1h_IIyy, hm_minimal.ms, axis=-1)
    IyIy_1h_at_i = np.trapz(itgnd_1h_IyIy, hm_minimal.ms, axis=-1)
    yIII_1h_at_i = np.trapz(itgnd_1h_yIII, hm_minimal.ms, axis=-1)

    oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)
    oneH_second_bispec_at_i = np.trapz(itgnd_1h_second_bispec, hm_minimal.ms, axis=-1)
    twoH_second_bispec_at_i = np.trapz(itgnd_2h_second_bispec, hm_minimal.ms, axis=-1)

    # Accumulate integrands for 1-3 2-halo trispectrum
    Iyyy_2h_1_3_at_i = np.trapz(itgnd_2h_Iyyy, hm_minimal.ms, axis=-1)
    IIyy_2h_1_3_at_i = np.trapz(itgnd_2h_IIyy, hm_minimal.ms, axis=-1)
    IyIy_2h_1_3_at_i = np.trapz(itgnd_2h_IyIy, hm_minimal.ms, axis=-1)
    yIII_2h_1_3_at_i = np.trapz(itgnd_2h_yIII, hm_minimal.ms, axis=-1)

    # Accumulate integrands for 2-2 2-halo trispectrum
    Iyyy_2h_2_2_at_i = 2 * np.trapz(itgnd_2h_Iy, hm_minimal.ms, axis=-1) \
                          * np.trapz(itgnd_2h_yy, hm_minimal.ms, axis=-1) * pk_of_L
    IIyy_2h_2_2_at_i = 2 * np.trapz(itgnd_2h_II, hm_minimal.ms, axis=-1) \
                          * np.trapz(itgnd_2h_yy, hm_minimal.ms, axis=-1) * pk_of_L
    IyIy_2h_2_2_at_i = 2 * np.trapz(itgnd_2h_Iy, hm_minimal.ms, axis=-1) ** 2 * pk_of_L
    yIII_2h_2_2_at_i = 2 * np.trapz(itgnd_2h_Iy, hm_minimal.ms, axis=-1) \
                          * np.trapz(itgnd_2h_II, hm_minimal.ms, axis=-1) * pk_of_L

    tmpCorr = np.trapz(itgnd_2h_k, hm_minimal.ms, axis=-1)
    twoH_cross_at_i = np.trapz(itgnd_2h_Iy, hm_minimal.ms, axis=-1) * (tmpCorr + hm_minimal.m_consistency[i]) * pk_of_L \
                      + np.trapz(itgnd_2h_kinHaloWfg, hm_minimal.ms, axis=-1)
    return Iyyy_1h_at_i, IIyy_1h_at_i, IyIy_1h_at_i, yIII_1h_at_i, oneH_cross_at_i, oneH_second_bispec_at_i, \
           twoH_second_bispec_at_i, Iyyy_2h_1_3_at_i, IIyy_2h_1_3_at_i, IyIy_2h_1_3_at_i, yIII_2h_1_3_at_i,\
           Iyyy_2h_2_2_at_i, IIyy_2h_2_2_at_i, IyIy_2h_2_2_at_i, yIII_2h_2_2_at_i, twoH_cross_at_i

def mixed_cross_itgrnds_each_z(i, ells_out, fftlog_way, damp_1h_prof, exp_minimal, hm_minimal, survey_name):
    """
    Obtain the integrand at the i-th redshift by doing the integrals over mass and the QE reconstructions.
    Input:
        * i = int. Index of the ith redshift in the halo model calculation
        * fftlog_way = Boolean. If true, use 1D fftlog reconstructions, otherwise use 2D quicklens
        * damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        * exp_minimal = instance of qest.exp_minimal(exp)
        * hm_minimal = instance of biases.hm_minimal(hm_framework)
    """
    print(f'Now in parallel loop {i}')
    nx = hm_minimal.lmax_out + 1 if fftlog_way else exp_minimal.pix.nx
    # Temporary storage
    itgnd_1h_cross = np.zeros([nx, hm_minimal.nMasses]) + 0j if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) + 0j;
    itgnd_2h_k = itgnd_1h_cross.copy(); itgnd_2h_II = itgnd_1h_cross.copy()
    itgnd_2h_kinHaloWfg = itgnd_1h_cross.copy();
    itgnd_I_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) # TODO: not sure what to do here with nx for QL
    itgnd_y_for_2hbispec = np.zeros([exp_minimal.lmax + 1, hm_minimal.nMasses]) if fftlog_way else np.zeros([nx, nx, hm_minimal.nMasses]) # TODO: not sure what to do here with nx for QL


    # To keep QE calls tidy, define
    QE = lambda prof_1, prof_2 : qest.get_TT_qe(fftlog_way, ells_out, prof_1, exp_minimal.qe_norm,
                                           exp_minimal.pix, exp_minimal.lmax, exp_minimal.cltt_tot, exp_minimal.ls,
                                           exp_minimal.cl_len.cltt, exp_minimal.qest_lib, exp_minimal.ivf_lib, prof_2,
                                           weights_mat_total=exp_minimal.weights_mat_total, nodes=exp_minimal.nodes)

    # Project the matter power spectrum for two-halo terms
    pk_of_l = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=exp_minimal.lmax)
    pk_of_L = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks, hm_minimal.Pzk[i], ellmax=hm_minimal.lmax_out)
    if not fftlog_way:
        pk_of_l = ql.spec.cl2cfft(pk_of_l, exp_minimal.pix).fft
        pk_of_L = ql.spec.cl2cfft(pk_of_L, exp_minimal.pix).fft

    # Integral over M for 2halo trispectrum. This will later go into a QE
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        # project the profiles
        y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                      hm_minimal.pk_profiles['y'][i, j], ellmax=exp_minimal.lmax)
        u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                         hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
        u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u

        itgnd_I_for_2hbispec[..., j] = (u_cen + u_sat) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_y_for_2hbispec[..., j] = y * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

    int_over_M_of_I = pk_of_l * (np.trapz(itgnd_I_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.I_consistency[i])
    int_over_M_of_y = pk_of_l * (np.trapz(itgnd_y_for_2hbispec, hm_minimal.ms, axis=-1) + hm_minimal.y_consistency[i])

    # M integral.
    for j, m in enumerate(hm_minimal.ms):
        if m > exp_minimal.massCut: continue
        y = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                                      hm_minimal.ks, hm_minimal.pk_profiles['y'][i, j], ellmax=exp_minimal.lmax)
        # project the galaxy profiles
        u = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                         hm_minimal.uk_profiles['nfw'][i, j], ellmax=exp_minimal.lmax)
        u_cen = hm_minimal.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
        u_sat = hm_minimal.CIB_satellite_filter[:, i, j] * u
        # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
        # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 # TODO:Why?
        gal = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                           hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j], ellmax=hm_minimal.lmax_out)
        galfft = gal / hm_minimal.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal, exp_minimal.pix).fft / \
                                                                            hm_minimal.hods[survey_name]['ngal'][i]

        phicfft_ucen_y = QE(u_cen, y)
        phicfft_usat_y = QE(u_sat, y)
        phicfft_I_intofy = QE(u_cen + u_sat, int_over_M_of_y)
        phicfft_y_intofI = QE(y, int_over_M_of_I)

        if damp_1h_prof:
            y_damp = exp_minimal.tsz_filter * tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                                               hm_minimal.ks, hm_minimal.pk_profiles['y'][i, j]
                                               * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=exp_minimal.lmax)
            # project the galaxy profiles
            u_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i], hm_minimal.ks,
                                  hm_minimal.uk_profiles['nfw'][i, j]
                                  * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=exp_minimal.lmax)
            u_sat_damp = hm_minimal.CIB_satellite_filter[:, i, j] * u_damp
            gal_damp = tls.pkToPell(hm_minimal.comoving_radial_distance[i],
                                    hm_minimal.ks, hm_minimal.uk_profiles['nfw'][i, j]
                                    * (1 - np.exp(-(hm_minimal.ks / hm_minimal.p['kstar_damping']))), ellmax=hm_minimal.lmax_out)
            galfft_damp = gal_damp / hm_minimal.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal_damp,
                                                                                                          exp_minimal.pix).fft / \
                                                                                          hm_minimal.hods[survey_name][
                                                                                              'ngal'][i]
            phicfft_ucen_y_damp = QE(u_cen, y_damp)
            phicfft_usat_y_damp = QE(u_sat_damp, y_damp)
        else:
            galfft_damp = galfft;
            phicfft_ucen_y_damp = phicfft_ucen_y;
            phicfft_usat_y_damp = phicfft_usat_y

        # Accumulate the itgnds
        mean_Ngal = hm_minimal.hods[survey_name]['Nc'][i, j] + hm_minimal.hods[survey_name]['Ns'][i, j]
        itgnd_1h_cross[..., j] = mean_Ngal * np.conjugate(galfft_damp) * hm_minimal.nzm[i, j] \
                                 * (phicfft_ucen_y_damp + phicfft_usat_y_damp)
        itgnd_2h_k[..., j] = mean_Ngal * np.conjugate(galfft) * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]
        itgnd_2h_II[..., j] = hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j] \
                              * (phicfft_ucen_y + phicfft_usat_y)
        itgnd_2h_kinHaloWfg[..., j] = mean_Ngal * np.conjugate(galfft) * ( phicfft_I_intofy + phicfft_y_intofI)\
                                * hm_minimal.nzm[i, j] * hm_minimal.bh_ofM[i, j]

    oneH_cross_at_i = np.trapz(itgnd_1h_cross, hm_minimal.ms, axis=-1)

    tmpCorr = np.trapz(itgnd_2h_k, hm_minimal.ms, axis=-1)
    twoH_cross_at_i = np.trapz(itgnd_2h_II, hm_minimal.ms, axis=-1) * (tmpCorr + hm_minimal.g_consistency[i]) * pk_of_L \
                      + np.trapz(itgnd_2h_kinHaloWfg, hm_minimal.ms, axis=-1)
    return oneH_cross_at_i, twoH_cross_at_i
