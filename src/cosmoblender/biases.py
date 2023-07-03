'''
Terminology:
itgnd = integrand
oneH = one halo
twoH = two halo
'''

import numpy as np
import hmvec as hm
from . import tools as tls
from . import second_bispec_bias_stuff as sbbs #TODO:remove this when the secondary bispectrum bias is properly incorporated
import quicklens as ql

class hm_framework:
    """ Set the halo model parameters """
    def __init__(self, lmax_out=3000, m_min=1e12, m_max=5e16, nMasses=30, z_min=0.07, z_max=3, nZs=30, k_min = 1e-4,\
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
                     tsz_consistency=False):
        """
        Calculate the tsz biases to the CMB lensing auto-spectrum (C^{\phi\phi}_L)
        given an "experiment" object (defined in qest.py)
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
        """
        hcos = self.hcos
        self.get_matter_consistency(exp)
        self.get_tsz_consistency(exp, lmax_proj=exp.lmax)
        if not tsz_consistency:
            self.y_consistency = np.zeros_like(self.y_consistency)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.pix.nx

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        tsz_filter = exp.get_tsz_filter()

        # The one and two halo bias terms -- these store the itgnd to be integrated over z.
        # Dimensions depend on method
        oneH_4pt = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        twoH_1_3 = oneH_4pt.copy(); twoH_2_2 = oneH_4pt.copy(); oneH_cross = oneH_4pt.copy();
        twoH_cross = oneH_4pt.copy();

        if get_secondary_bispec_bias:
            # TODO: choose an Lmin that makes sense given Limber
            lbins_sec_bispec_bias = np.arange(10, self.lmax_out + 1, bin_width_out_second_bispec_bias)
            oneH_second_bispec = np.zeros([len(lbins_sec_bispec_bias),self.nZs])+0j
            # Get QE normalisation
            qe_norm_1D = exp.qe_norm.get_ml(np.arange(10, self.lmax_out, 40))

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_4pt = np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            itgnd_1h_cross = itgnd_1h_4pt.copy();  itgnd_2h_1_3_trispec = itgnd_1h_4pt.copy();
            itgnd_2h_2_2_trispec = itgnd_1h_4pt.copy(); itgnd_2h_1g = itgnd_1h_4pt.copy();
            itgnd_2h_2g = itgnd_1h_4pt.copy();
            integ_1h_for_2htrispec = np.zeros([nx,self.nMasses]) if fftlog_way else np.zeros([nx,nx,self.nMasses])

            if get_secondary_bispec_bias:
                itgnd_1h_second_bispec = np.zeros([len(lbins_sec_bispec_bias),self.nMasses])+0j

            # Project the matter power spectrum for two-halo terms
            pk_of_l = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=exp.lmax)
            pk_of_L = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk_of_l = ql.spec.cl2cfft(pk_of_l, exp.pix).fft
                pk_of_L = ql.spec.cl2cfft(pk_of_L, exp.pix).fft

            # Integral over M for 2halo trispectrum. This will later go into a QE
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                y = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                              hcos.pk_profiles['y'][i,j], ellmax=exp.lmax)
                integ_1h_for_2htrispec[..., j] = y * hcos.nzm[i, j] * hcos.bh_ofM[i, j]

            # Do the 1- integral in the 1-3 trispectrum and impose consistency condition
            int_over_M_of_profile = pk_of_l * (np.trapz(integ_1h_for_2htrispec,hcos.ms,axis=-1) + self.y_consistency[i])

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                y = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                              hcos.pk_profiles['y'][i,j], ellmax=exp.lmax)
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks, hcos.uk_profiles['nfw'][i,j], ellmax=self.lmax_out)

                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]

                phicfft = exp.get_TT_qe(fftlog_way, ells_out, y,y)
                phicfft_mixed = exp.get_TT_qe(fftlog_way, ells_out, y, int_over_M_of_profile)

                # Consider damping the profiles at low k in 1h terms to avoid it exceeding many-halo amplitude
                if damp_1h_prof:
                    y_damp = tsz_filter * \
                             tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                          hcos.pk_profiles['y'][i,j]*(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                          ellmax=exp.lmax)
                    kap_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                       hcos.ks, hcos.uk_profiles['nfw'][i, j], ellmax=self.lmax_out)
                    kfft_damp = kap_damp * self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp, exp.pix).fft * self.ms_rescaled[j]
                    phicfft_damp = exp.get_TT_qe(fftlog_way, ells_out, y_damp, y_damp)
                else:
                    y_damp = y; kfft_damp = kfft; phicfft_damp = phicfft

                # Accumulate the itgnds
                itgnd_1h_cross[...,j] = phicfft_damp*np.conjugate(kfft_damp)*hcos.nzm[i,j]
                itgnd_1h_4pt[...,j] = phicfft_damp*np.conjugate(phicfft_damp) * hcos.nzm[i,j]
                itgnd_2h_1g[...,j] = np.conjugate(kfft)*hcos.nzm[i,j]*hcos.bh_ofM[i,j]
                itgnd_2h_2g[..., j] = phicfft * hcos.nzm[i, j] * hcos.bh_ofM[i, j]
                itgnd_2h_1_3_trispec[...,j] = phicfft*np.conjugate(phicfft_mixed)*hcos.nzm[i,j]*hcos.bh_ofM[i,j]
                itgnd_2h_2_2_trispec[...,j] = phicfft*hcos.nzm[i,j]*hcos.bh_ofM[i,j]

                if get_secondary_bispec_bias:
                    # Temporary secondary bispectrum bias stuff
                    # The part with the nested lensing reconstructions
                    exp_param_dict = {'lmax': exp.lmax, 'nx': exp.nx, 'dx_arcmin': exp.dx*60.*180./np.pi}
                    # Get the kappa map, up to lmax rather than lmax_out as was needed in other terms
                    if damp_1h_prof:
                        kap_secbispec = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                     hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                    else:
                        kap_secbispec = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                     hcos.uk_profiles['nfw'][i, j]
                                                     *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                                     ellmax=exp.lmax)
                    sec_bispec_rec = sbbs.get_sec_bispec_bias(lbins_sec_bispec_bias, qe_norm_1D,
                                                              exp_param_dict, exp.cltt_tot,
                                                              y_damp, kap_secbispec*self.ms_rescaled[j],
                                                              parallelise=parallelise_secondbispec)
                    itgnd_1h_second_bispec[..., j] = hcos.nzm[i,j] * sec_bispec_rec
                    # TODO:add the 2-halo term. Should be easy.
            # Perform the m integrals
            oneH_4pt[...,i]=np.trapz(itgnd_1h_4pt,hcos.ms,axis=-1)
            oneH_cross[...,i]=np.trapz(itgnd_1h_cross,hcos.ms,axis=-1)
            if get_secondary_bispec_bias:
                oneH_second_bispec[...,i]=np.trapz(itgnd_1h_second_bispec,hcos.ms,axis=-1)

            # TODO: apply consistency to 2-2 trispectrum
            twoH_2_2[...,i]= 2 * np.trapz(itgnd_2h_2_2_trispec,hcos.ms,axis=-1)**2 * pk_of_L
            twoH_1_3[...,i]= 4 * np.trapz(itgnd_2h_1_3_trispec,hcos.ms,axis=-1)
            tmpCorr =np.trapz(itgnd_2h_1g,hcos.ms,axis=-1)
            twoH_cross[...,i]= np.trapz(itgnd_2h_2g,hcos.ms,axis=-1) * (tmpCorr + self.m_consistency[i]) * pk_of_L

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
            exp.biases['second_bispec_bias_ells'] = lbins_sec_bispec_bias

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['tsz']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return

    def get_tsz_cross_biases(self, exp, gzs, gdndz, fftlog_way=True, bin_width_out=30, survey_name='LSST',
                             damp_1h_prof=True, gal_consistency=False):
        """
        Calculate the tsz biases to the cross-correlation of CMB lensing with a galaxy survey,
        given an "experiment" object (defined in qest.py)
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
        """
        hcos = self.hcos
        self.get_galaxy_consistency(exp, survey_name)
        if not gal_consistency:
            self.g_consistency = np.zeros_like(self.g_consistency)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.pix.nx

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        tsz_filter = exp.get_tsz_filter()

        # The one and two halo bias terms -- these store the itgnd to be integrated over z.
        # Dimensions depend on method
        oneH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        twoH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_cross =np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            itgnd_2h_2g = np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            itgnd_2h_1g = np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                y = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,
                                              hcos.pk_profiles['y'][i,j], ellmax=exp.lmax)
                # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
                # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 #TODO: why do you say that?
                gal = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks,hcos.uk_profiles['nfw'][i,j], ellmax=self.lmax_out)
                #TODO: should ngal in denominator depend on z? ms_rescaled doesn't
                galfft = gal / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal, exp.pix).fft / hcos.hods[survey_name]['ngal'][i]
                phicfft = exp.get_TT_qe(fftlog_way, ells_out, y, y)

                # Consider damping the profiles at low k in 1h terms to avoid it exceeding many-halo amplitude
                if damp_1h_prof:
                    y_damp = tsz_filter\
                             * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                            hcos.pk_profiles['y'][i, j]
                                            *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=exp.lmax)
                    gal_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                            hcos.uk_profiles['nfw'][i, j]
                                            *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                    galfft_damp = gal_damp / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal_damp, exp.pix).fft / hcos.hods[survey_name]['ngal'][i]
                    phicfft_damp = exp.get_TT_qe(fftlog_way, ells_out, y_damp, y_damp)
                else:
                    y_damp = y; gal_damp = gal; galfft_damp = galfft; phicfft_damp = phicfft

                # Accumulate the itgnds
                mean_Ngal = hcos.hods[survey_name]['Nc'][i, j] + hcos.hods[survey_name]['Ns'][i, j]
                itgnd_1h_cross[..., j] = mean_Ngal * phicfft_damp * np.conjugate(galfft_damp) * hcos.nzm[i, j]
                itgnd_2h_1g[..., j] = mean_Ngal * np.conjugate(galfft) * hcos.nzm[i, j] * hcos.bh_ofM[i, j]
                itgnd_2h_2g[..., j] = phicfft * hcos.nzm[i, j] * hcos.bh_ofM[i, j]

            # Perform the m integrals
            oneH_cross[...,i]=np.trapz(itgnd_1h_cross,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            tmpCorr =np.trapz(itgnd_2h_1g,hcos.ms,axis=-1)
            twoH_cross[...,i] = np.trapz(itgnd_2h_2g,hcos.ms,axis=-1) * (tmpCorr + self.g_consistency[i]) * pk
            twoH_cross[...,i] = np.trapz(itgnd_2h_2g,hcos.ms,axis=-1) * (tmpCorr + self.g_consistency[i]) * pk

        # Integrate over z
        gyy_intgrnd = self.T_CMB ** 2 * tls.limber_itgrnd_kernel(hcos, 3) \
                      * tls.gal_window(hcos, hcos.zs, gzs, gdndz) * tls.y_window(hcos) ** 2
        exp.biases['tsz']['cross_w_gals']['1h'] = np.trapz(oneH_cross * gyy_intgrnd, hcos.zs, axis=-1)
        exp.biases['tsz']['cross_w_gals']['2h'] = np.trapz(twoH_cross * gyy_intgrnd, hcos.zs, axis=-1)
        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['tsz']['cross_w_gals']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['cross_w_gals']['1h']).get_ml(lbins).specs['cl']
            exp.biases['tsz']['cross_w_gals']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['tsz']['cross_w_gals']['2h']).get_ml(lbins).specs['cl']
            return

    def get_tsz_ps(self, exp, damp_1h_prof=True, gal_consistency=False):
        """
        Calculate the tSZ power spectrum
        Input:
            * exp = a qest.experiment object
            * (optional) damp_1h_prof = Bool. Default is False. Whether to damp the profiles at low k in 1h terms
        """
        hcos = self.hcos

        # Output ells
        ells_out = np.arange(self.lmax_out+1)

        nx = self.lmax_out+1

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_ps_tz = np.zeros([nx,self.nZs])+0j
        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_ps_tSZ = np.zeros([nx,self.nMasses])+0j

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                if damp_1h_prof:
                    y = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                     hcos.ks,hcos.pk_profiles['y'][i,j]*(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                     ellmax=self.lmax_out)
                else:
                    y = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.pk_profiles['y'][i,j],
                                     ellmax=self.lmax_out)
                # Accumulate the itgnds
                itgnd_1h_ps_tSZ[:,j] = y*np.conjugate(y)*hcos.nzm[i,j]

                # Perform the m integrals
            oneH_ps_tz[:,i]=np.trapz(itgnd_1h_ps_tSZ,hcos.ms,axis=-1)

        # Integrate over z
        yy_integrand = self.T_CMB ** 2 * tls.limber_itgrnd_kernel(hcos, 2) * tls.y_window(hcos) ** 2
        ps_oneH_tSZ = np.trapz( oneH_ps_tz * yy_integrand, hcos.zs, axis=-1)

        # ToDo: implement 2 halo term for tSZ PS tests
        ps_twoH_tSZ = np.zeros(ps_oneH_tSZ.shape)
        return ps_oneH_tSZ, ps_twoH_tSZ

    def get_g_cross_kappa(self, exp, survey_name, gzs, gdndz, damp_1h_prof=True, fftlog_way=True):
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
        self.get_galaxy_consistency(exp, survey_name)
        if not gal_consistency:
            self.g_consistency = np.zeros_like(self.g_consistency)
        self.get_matter_consistency(exp)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)

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
                     cib_consistency=False):
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
        """
        hcos = self.hcos

        # Pre-calculate consistency for 2h integrals
        self.get_matter_consistency(exp)
        self.get_cib_consistency(exp, lmax_proj=exp.lmax)
        if not cib_consistency:
            self.I_consistency = np.zeros_like(self.I_consistency)

        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)
        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        IIII_1h = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        IIII_2h_1_3 = IIII_1h.copy(); IIII_2h_2_2 = IIII_1h.copy(); oneH_cross = IIII_1h.copy();
        twoH_cross = IIII_1h.copy()

        if get_secondary_bispec_bias:
            # TODO: choose an Lmin that makes sense given Limber
            lbins_sec_bispec_bias = np.arange(10, self.lmax_out + 1, bin_width_out_second_bispec_bias)
            oneH_second_bispec = np.zeros([len(lbins_sec_bispec_bias),self.nZs])+0j
            # Get QE normalisation
            qe_norm_1D = exp.qe_norm.get_ml(np.arange(10, self.lmax_out, 40))

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_cross=np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            itgnd_1h_IIII=itgnd_1h_cross.copy(); itgnd_2h_k=itgnd_1h_cross.copy()
            itgnd_2h_II=itgnd_1h_cross.copy(); itgnd_2h_IintIII=itgnd_1h_cross.copy()
            integ_1h_for_2htrispec=np.zeros([nx,self.nMasses]) if fftlog_way else np.zeros([nx,nx,self.nMasses])

            if get_secondary_bispec_bias:
                itgnd_1h_second_bispec = np.zeros([len(lbins_sec_bispec_bias),self.nMasses])+0j

            # Project the matter power spectrum for two-halo terms
            pk_of_l = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=exp.lmax)
            pk_of_L = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk_of_l = ql.spec.cl2cfft(pk_of_l, exp.pix).fft
                pk_of_L = ql.spec.cl2cfft(pk_of_L, exp.pix).fft

            # Integral over M for 2halo trispectrum. This will later go into a QE
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                          hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                u_cen = self.CIB_central_filter[:,i,j] # Centrals come with a factor of u^0
                u_sat = self.CIB_satellite_filter[:,i,j] * u

                integ_1h_for_2htrispec[..., j] = hcos.nzm[i, j] * hcos.bh_ofM[i, j] * (u_cen + u_sat)

            # Do the 1- integral in the 1-3 trispectrum and impose consistency condition
            Iint = pk_of_l * (np.trapz(integ_1h_for_2htrispec,hcos.ms,axis=-1) + self.I_consistency[i])

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                          hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                u_cen = self.CIB_central_filter[:,i,j] # Centrals come with a factor of u^0
                u_sat = self.CIB_satellite_filter[:,i,j] * u

                phicfft_ucen_usat = exp.get_TT_qe(fftlog_way, ells_out, u_cen, u_sat)
                phicfft_usat_usat = exp.get_TT_qe(fftlog_way, ells_out, u_sat, u_sat)
                phicfft_Iint_ucen = exp.get_TT_qe(fftlog_way, ells_out, Iint, u_cen)
                phicfft_Iint_usat = exp.get_TT_qe(fftlog_way, ells_out, Iint, u_sat)

                # Get the kappa map
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks,hcos.uk_profiles['nfw'][i,j], ellmax=self.lmax_out)
                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]

                if damp_1h_prof:
                    # Damp the profiles on large scales when calculating 1h terms
                    # Note that we are damping u_sat, but leaving u_cen as is, because it's always at the center
                    u_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                     hcos.uk_profiles['nfw'][i, j]*(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                          ellmax=exp.lmax)
                    # TODO: check that the correct way to damp u_cen is to leave it as is
                    u_sat_damp = self.CIB_satellite_filter[:, i, j] * u_damp

                    phicfft_ucen_usat_damp = exp.get_TT_qe(fftlog_way, ells_out, u_cen, u_sat_damp)
                    phicfft_usat_usat_damp = exp.get_TT_qe(fftlog_way, ells_out, u_sat_damp, u_sat_damp)

                    # Get the kappa map
                    kap_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                       hcos.ks, hcos.uk_profiles['nfw'][i, j]
                                            *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                    kfft_damp = kap_damp * self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp, exp.pix).fft * self.ms_rescaled[j]
                else:
                    u_sat_damp=u_sat; phicfft_ucen_usat_damp=phicfft_ucen_usat;
                    phicfft_usat_usat_damp=phicfft_usat_usat; kfft_damp=kfft

                # Accumulate the itgnds
                itgnd_1h_cross[...,j] = hcos.nzm[i,j] * np.conjugate(kfft_damp) * (phicfft_usat_usat_damp +
                                                                                       2 * phicfft_ucen_usat_damp)
                itgnd_1h_IIII[...,j] = hcos.nzm[i,j] * (phicfft_usat_usat_damp * np.conjugate(phicfft_usat_usat_damp)
                                                        + 4*phicfft_ucen_usat_damp*np.conjugate(phicfft_usat_usat_damp))

                itgnd_2h_k[...,j] = np.conjugate(kfft) * hcos.nzm[i,j] * hcos.bh_ofM[i,j]
                itgnd_2h_II[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] * (phicfft_usat_usat + 2 * phicfft_ucen_usat)
                itgnd_2h_IintIII[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j]\
                                          * (phicfft_Iint_ucen * np.conjugate(phicfft_usat_usat)
                                             +phicfft_Iint_usat * np.conjugate(2*phicfft_ucen_usat + phicfft_usat_usat))
                itgnd_2h_II[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] * (phicfft_usat_usat + 2 * phicfft_ucen_usat)


                if get_secondary_bispec_bias:
                    # Temporary secondary bispectrum bias stuff
                    # The part with the nested lensing reconstructions
                    exp_param_dict = {'lmax': exp.lmax, 'nx': exp.nx, 'dx_arcmin': exp.dx*60.*180./np.pi}
                    # Get the kappa map, up to lmax rather than lmax_out as was needed in other terms
                    if damp_1h_prof:
                        kap_secbispec = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                     hcos.uk_profiles['nfw'][i, j]
                                                     *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                                     ellmax=exp.lmax)
                    else:
                        kap_secbispec = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                     hcos.uk_profiles['nfw'][i, j],
                                                     ellmax=exp.lmax)
                    sec_bispec_rec = 2 * sbbs.get_sec_bispec_bias(lbins_sec_bispec_bias, qe_norm_1D,
                                                                  exp_param_dict, exp.cltt_tot, u_cen,
                                                                  kap_secbispec*self.ms_rescaled[j],
                                                                  projected_fg_profile_2 = u_sat_damp,
                                                                  parallelise=parallelise_secondbispec)\
                                     + sbbs.get_sec_bispec_bias(lbins_sec_bispec_bias, qe_norm_1D, exp_param_dict,
                                                                exp.cltt_tot, u_sat_damp,
                                                                kap_secbispec*self.ms_rescaled[j],
                                                                projected_fg_profile_2 = u_sat_damp,
                                                                parallelise=parallelise_secondbispec)
                    itgnd_1h_second_bispec[..., j] = hcos.nzm[i,j] * sec_bispec_rec
                    # TODO:add the 2-halo term. Should be easy.

            # Perform the m integrals
            IIII_1h[...,i]=np.trapz(itgnd_1h_IIII,hcos.ms,axis=-1)

            oneH_cross[...,i]=np.trapz(itgnd_1h_cross,hcos.ms,axis=-1)

            if get_secondary_bispec_bias:
                oneH_second_bispec[...,i]=np.trapz(itgnd_1h_second_bispec,hcos.ms,axis=-1)

            IIII_2h_1_3[...,i] = 4 * np.trapz(itgnd_2h_IintIII,hcos.ms,axis=-1)
            # TODO: implement consistency for 2-2 trispectrum
            IIII_2h_2_2[..., i] = 2 * np.trapz(itgnd_2h_II, hcos.ms, axis=-1) ** 2 * pk_of_L

            tmpCorr =np.trapz(itgnd_2h_k,hcos.ms,axis=-1)
            twoH_cross[...,i]=np.trapz(itgnd_2h_II,hcos.ms,axis=-1)\
                                 *(tmpCorr + self.m_consistency[i]) * pk_of_L

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
            exp.biases['cib']['second_bispec']['1h'] = np.trapz( oneH_second_bispec * kII_itgnd, hcos.zs, axis=-1)
            exp.biases['second_bispec_bias_ells'] = lbins_sec_bispec_bias

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['cib']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
            return

    def get_cib_cross_biases(self, exp, gzs, gdndz, fftlog_way=True, bin_width_out=30, survey_name='LSST',
                             damp_1h_prof=True, gal_consistency=False):
        """
        Calculate the CIB biases to the cross-correlation of CMB lensing with a galaxy survey,
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
        """
        hcos = self.hcos
        self.get_galaxy_consistency(exp, survey_name)
        if not gal_consistency:
            self.g_consistency = np.zeros_like(self.g_consistency)

        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j;
        twoH_cross = oneH_cross.copy()

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_cross=np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j;
            itgnd_2h_k=itgnd_1h_cross.copy(); itgnd_2h_II=itgnd_1h_cross.copy()

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                 hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                u_cen = self.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
                u_sat = self.CIB_satellite_filter[:, i, j] * u

                # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
                # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 # TODO:Why?
                gal = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks,hcos.uk_profiles['nfw'][i,j], ellmax=self.lmax_out)
                galfft = gal / hcos.hods[survey_name]['ngal'][i]  if fftlog_way else ql.spec.cl2cfft(gal, exp.pix).fft / hcos.hods[survey_name]['ngal'][i]

                phicfft_ucen_usat = exp.get_TT_qe(fftlog_way, ells_out, u_cen, u_sat)
                phicfft_usat_usat = exp.get_TT_qe(fftlog_way, ells_out, u_sat, u_sat)

                if damp_1h_prof:
                    u_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                     hcos.uk_profiles['nfw'][i, j]*(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                          ellmax=exp.lmax)
                    u_sat_damp = self.CIB_satellite_filter[:, i, j] * u_damp
                    gal_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                       hcos.ks, hcos.uk_profiles['nfw'][i, j]
                                            *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                    galfft_damp = gal_damp / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal_damp, exp.pix).fft / hcos.hods[survey_name]['ngal'][i]

                    phicfft_ucen_usat_damp = exp.get_TT_qe(fftlog_way, ells_out, u_cen, u_sat_damp)
                    phicfft_usat_usat_damp = exp.get_TT_qe(fftlog_way, ells_out, u_sat_damp, u_sat_damp)
                else:
                    galfft_damp=galfft;phicfft_ucen_usat_damp=phicfft_ucen_usat;
                    phicfft_usat_usat_damp=phicfft_usat_usat

                # Accumulate the itgnds
                mean_Ngal = hcos.hods[survey_name]['Nc'][i, j] + hcos.hods[survey_name]['Ns'][i, j]
                itgnd_1h_cross[...,j] = mean_Ngal * np.conjugate(galfft_damp) * hcos.nzm[i,j] *\
                                                 (phicfft_usat_usat_damp + 2 * phicfft_ucen_usat_damp)
                itgnd_2h_k[...,j] = mean_Ngal * np.conjugate(galfft) * hcos.nzm[i,j] * hcos.bh_ofM[i,j]
                itgnd_2h_II[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] \
                                              * ((phicfft_usat_usat + 2 * phicfft_ucen_usat))

            oneH_cross[...,i]=np.trapz(itgnd_1h_cross,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            tmpCorr =np.trapz(itgnd_2h_k,hcos.ms,axis=-1)
            twoH_cross[...,i] = np.trapz(itgnd_2h_II,hcos.ms,axis=-1) * (tmpCorr + self.g_consistency[i]) * pk


        # itgnd factors from Limber projection (adapted to hmvec conventions)
        gII_itgnd = tls.limber_itgrnd_kernel(hcos, 3) * tls.gal_window(hcos, hcos.zs, gzs, gdndz) * tls.CIB_window(hcos)**2

        # Integrate over z
        exp.biases['cib']['cross_w_gals']['1h'] = np.trapz( oneH_cross*gII_itgnd, hcos.zs, axis=-1)
        exp.biases['cib']['cross_w_gals']['2h'] = np.trapz( twoH_cross*gII_itgnd, hcos.zs, axis=-1)

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['cib']['cross_w_gals']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['cross_w_gals']['1h']).get_ml(lbins).specs['cl']
            exp.biases['cib']['cross_w_gals']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['cross_w_gals']['2h']).get_ml(lbins).specs['cl']
            return

    def get_mixed_cross_biases(self, exp, gzs, gdndz, fftlog_way=True, bin_width_out=30, survey_name='LSST',
                               damp_1h_prof=True, gal_consistency=False):
        """
        Calculate the mixed tsz-cib  biases to the cross-correlation of CMB lensing with a galaxy survey,
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
        """
        hcos = self.hcos
        self.get_galaxy_consistency(exp, survey_name)
        if not gal_consistency:
            self.g_consistency = np.zeros_like(self.g_consistency)
        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)
        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        tsz_filter = exp.get_tsz_filter()

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        oneH_cross = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j;
        twoH_cross = oneH_cross.copy()

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_cross=np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j;
            itgnd_2h_k=itgnd_1h_cross.copy(); itgnd_2h_II=itgnd_1h_cross.copy()

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                y = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                              hcos.ks,hcos.pk_profiles['y'][i,j], ellmax=exp.lmax)
                #project the galaxy profiles
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                 hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                u_cen = self.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
                u_sat = self.CIB_satellite_filter[:, i, j] * u
                # Get the galaxy map --- analogous to kappa in the auto-biases. Note that we need a factor of
                # H dividing the galaxy window function to translate the hmvec convention to e.g. Ferraro & Hill 18 # TODO:Why?
                gal = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                   hcos.ks,hcos.uk_profiles['nfw'][i,j], ellmax=self.lmax_out)
                galfft = gal / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal, exp.pix).fft / hcos.hods[survey_name]['ngal'][i]

                phicfft_ucen_y = exp.get_TT_qe(fftlog_way, ells_out, u_cen, y)
                phicfft_usat_y = exp.get_TT_qe(fftlog_way, ells_out, u_sat, y)

                if damp_1h_prof:
                    y_damp = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                                  hcos.ks, hcos.pk_profiles['y'][i, j]
                                                  *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=exp.lmax)
                    # project the galaxy profiles
                    u_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                     hcos.uk_profiles['nfw'][i, j]
                                     *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=exp.lmax)
                    u_sat_damp = self.CIB_satellite_filter[:, i, j] * u_damp
                    gal_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),
                                       hcos.ks, hcos.uk_profiles['nfw'][i, j]
                                       *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                    galfft_damp = gal_damp / hcos.hods[survey_name]['ngal'][i] if fftlog_way else ql.spec.cl2cfft(gal_damp, exp.pix).fft / hcos.hods[survey_name]['ngal'][i]
                    phicfft_ucen_y_damp = exp.get_TT_qe(fftlog_way, ells_out, u_cen, y_damp)
                    phicfft_usat_y_damp = exp.get_TT_qe(fftlog_way, ells_out, u_sat_damp, y_damp)
                else:
                    galfft_damp=galfft; phicfft_ucen_y_damp=phicfft_ucen_y; phicfft_usat_y_damp=phicfft_usat_y

                # Accumulate the itgnds
                mean_Ngal = hcos.hods[survey_name]['Nc'][i, j] + hcos.hods[survey_name]['Ns'][i, j]
                itgnd_1h_cross[...,j] =  mean_Ngal * np.conjugate(galfft_damp) * hcos.nzm[i,j] \
                                                  * (phicfft_ucen_y_damp + phicfft_usat_y_damp)
                itgnd_2h_k[...,j] = mean_Ngal * np.conjugate(galfft) * hcos.nzm[i,j] * hcos.bh_ofM[i,j]
                itgnd_2h_II[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] \
                                              * (phicfft_ucen_y + phicfft_usat_y)

            oneH_cross[...,i]=np.trapz(itgnd_1h_cross,hcos.ms,axis=-1)

            # This is the two halo term. P_k times the M integrals
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk = ql.spec.cl2cfft(pk, exp.pix).fft

            tmpCorr =np.trapz(itgnd_2h_k,hcos.ms,axis=-1)
            twoH_cross[...,i] = np.trapz(itgnd_2h_II,hcos.ms,axis=-1) * (tmpCorr + self.g_consistency[i]) * pk

        # itgnd factors from Limber projection (adapted to hmvec conventions)
        gIy_itgnd = 2 * tls.scale_sz(exp.freq_GHz) * self.T_CMB * tls.limber_itgrnd_kernel(hcos, 3)\
                    * tls.gal_window(hcos, hcos.zs, gzs, gdndz) * tls.CIB_window(hcos) * tls.y_window(hcos)

        # Integrate over z
        exp.biases['mixed']['cross_w_gals']['1h'] = np.trapz( oneH_cross*gIy_itgnd, hcos.zs, axis=-1)
        exp.biases['mixed']['cross_w_gals']['2h'] = np.trapz(twoH_cross*gIy_itgnd, hcos.zs, axis=-1)

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['cib']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['mixed']['cross_w_gals']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['cross_w_gals']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['cross_w_gals']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['cross_w_gals']['2h']).get_ml(lbins).specs['cl']
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
        self.get_cib_consistency(exp)
        if not cib_consistency:
            self.I_consistency = np.zeros_like(self.I_consistency)

        nx = self.lmax_out+1

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
            pk = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=self.lmax_out)
            twoH_ps[:, i] = (np.trapz(itgnd_2h_1g, hcos.ms, axis=-1) +self.I_consistency[i])** 2 * pk


        # Integrate over z
        II_itgrnd = tls.limber_itgrnd_kernel(hcos, 2) * tls.CIB_window(hcos) ** 2
        clCIBCIB_oneH_ps = np.trapz( oneH_ps * II_itgrnd, hcos.zs, axis=-1)
        clCIBCIB_twoH_ps = np.trapz( twoH_ps * II_itgrnd, hcos.zs, axis=-1)
        return clCIBCIB_oneH_ps, clCIBCIB_twoH_ps

    def get_mixed_auto_biases(self, exp, fftlog_way=True, get_secondary_bispec_bias=False, bin_width_out=30, \
                         bin_width_out_second_bispec_bias=250, parallelise_secondbispec=True, damp_1h_prof=True,
                         tsz_consistency=False, cib_consistency=False):
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
        """
        hcos = self.hcos
        # Get consistency conditions for 2h terms
        self.get_matter_consistency(exp)
        self.get_tsz_consistency(exp, lmax_proj=exp.lmax)
        self.get_cib_consistency(exp, lmax_proj=exp.lmax)
        if not tsz_consistency:
            self.y_consistency = np.zeros_like(self.y_consistency)
        if not cib_consistency:
            self.cib_consistency = np.zeros_like(self.cib_consistency)

        # Get frequency scaling of tSZ, possibly including harmonic ILC cleaning
        tsz_filter = exp.get_tsz_filter()
        # Compute effective CIB weights, including f_cen and f_sat factors as well as possibly fg cleaning
        self.get_CIB_filters(exp)

        # Output ells
        ells_out = np.arange(self.lmax_out+1)
        if not fftlog_way:
            lbins = np.arange(1,self.lmax_out+1,bin_width_out)

        nx = self.lmax_out+1 if fftlog_way else exp.nx

        # The one and two halo bias terms -- these store the itgnd to be integrated over z
        Iyyy_1h = np.zeros([nx,self.nZs])+0j if fftlog_way else np.zeros([nx,nx,self.nZs])+0j
        IIyy_1h = Iyyy_1h.copy(); yIII_1h = Iyyy_1h.copy(); Iyyy_2h_2_2 = Iyyy_1h.copy(); Iyyy_2h_1_3 = Iyyy_1h.copy();
        IIyy_2h_2_2 = Iyyy_1h.copy(); IIyy_2h_1_3 = Iyyy_1h.copy(); yIII_2h_1_3 = Iyyy_1h.copy();
        yIII_2h_2_2 = Iyyy_1h.copy(); oneH_cross = Iyyy_1h.copy(); twoH_cross = Iyyy_1h.copy();
        IyIy_2h_2_2 = Iyyy_1h.copy(); IyIy_2h_1_3 = Iyyy_1h.copy(); IyIy_1h = Iyyy_1h.copy()

        if get_secondary_bispec_bias:
            # TODO: choose an Lmin that makes sense given Limber
            lbins_sec_bispec_bias = np.arange(10, self.lmax_out + 1, bin_width_out_second_bispec_bias)
            oneH_second_bispec = np.zeros([len(lbins_sec_bispec_bias),self.nZs])+0j
            # Get QE normalisation
            qe_norm_1D = exp.qe_norm.get_ml(np.arange(10, self.lmax_out, 40))

        for i,z in enumerate(hcos.zs):
            #Temporary storage
            itgnd_1h_cross=np.zeros([nx,self.nMasses])+0j if fftlog_way else np.zeros([nx,nx,self.nMasses])+0j
            itgnd_1h_Iyyy=itgnd_1h_cross.copy(); itgnd_1h_IIyy=itgnd_1h_cross.copy()
            itgnd_1h_yIII=itgnd_1h_cross.copy(); itgnd_2h_k=itgnd_1h_cross.copy()
            itgnd_2h_Iy=itgnd_1h_cross.copy();itgnd_1h_IyIy=itgnd_1h_cross.copy()
            integ_1h_I_for_2htrispec = np.zeros([nx, self.nMasses]) if fftlog_way else np.zeros([nx,nx,self.nMasses])
            integ_1h_y_for_2htrispec = integ_1h_I_for_2htrispec.copy()
            itgnd_2h_Iyyy=itgnd_1h_cross.copy();itgnd_2h_IIyy=itgnd_1h_cross.copy();
            itgnd_2h_IyIy=itgnd_1h_cross.copy();itgnd_2h_yIII=itgnd_1h_cross.copy();
            itgnd_2h_Iy=itgnd_1h_cross.copy(); itgnd_2h_yy=itgnd_1h_cross.copy(); itgnd_2h_II=itgnd_1h_cross.copy();

            if get_secondary_bispec_bias:
                itgnd_1h_second_bispec = np.zeros([len(lbins_sec_bispec_bias),self.nMasses])+0j

            # Project the matter power spectrum for two-halo terms
            pk_of_l = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=exp.lmax)
            pk_of_L = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks, hcos.Pzk[i], ellmax=self.lmax_out)
            if not fftlog_way:
                pk_of_l = ql.spec.cl2cfft(pk_of_l, exp.pix).fft
                pk_of_L = ql.spec.cl2cfft(pk_of_L, exp.pix).fft

            # Integral over M for 2halo trispectrum. This will later go into a QE
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                          hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                y = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                              hcos.pk_profiles['y'][i, j], ellmax=exp.lmax)
                u_cen = self.CIB_central_filter[:,i,j] # Centrals come with a factor of u^0
                u_sat = self.CIB_satellite_filter[:,i,j] * u

                integ_1h_I_for_2htrispec[..., j] = hcos.nzm[i, j] * hcos.bh_ofM[i, j] * (u_cen + u_sat)
                integ_1h_y_for_2htrispec[..., j] = hcos.nzm[i, j] * hcos.bh_ofM[i, j] * y

            # Do the 1- integrals in the 1-3 trispectrum and impose consistency conditions
            Iint = pk_of_l * (np.trapz(integ_1h_I_for_2htrispec, hcos.ms, axis=-1) + self.I_consistency[i])
            yint = pk_of_l * (np.trapz(integ_1h_y_for_2htrispec, hcos.ms, axis=-1) + self.y_consistency[i])

            # M integral.
            for j,m in enumerate(hcos.ms):
                if m> exp.massCut: continue
                #project the galaxy profiles
                y = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                              hcos.pk_profiles['y'][i, j], ellmax=exp.lmax)
                u = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                 hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                u_cen = self.CIB_central_filter[:, i, j]  # Centrals come with a factor of u^0
                u_sat = self.CIB_satellite_filter[:, i, j] * u

                phicfft_ucen_usat = exp.get_TT_qe(fftlog_way, ells_out, u_cen, u_sat)
                phicfft_usat_usat = exp.get_TT_qe(fftlog_way, ells_out, u_sat, u_sat)
                phicfft_ucen_y = exp.get_TT_qe(fftlog_way, ells_out, u_cen, y)
                phicfft_usat_y = exp.get_TT_qe(fftlog_way, ells_out, u_sat, y)
                phicfft_yy = exp.get_TT_qe(fftlog_way, ells_out, y, y)

                phicfft_Iint_usat = exp.get_TT_qe(fftlog_way, ells_out, Iint, u_sat)
                phicfft_Iint_ucen = exp.get_TT_qe(fftlog_way, ells_out, Iint, u_cen)
                phicfft_Iint_y = exp.get_TT_qe(fftlog_way, ells_out, Iint, y)
                phicfft_yint_usat = exp.get_TT_qe(fftlog_way, ells_out, yint, u_sat)
                phicfft_yint_ucen = exp.get_TT_qe(fftlog_way, ells_out, yint, u_cen)
                phicfft_yint_y = exp.get_TT_qe(fftlog_way, ells_out, yint, y)

                # Get the kappa map
                kap = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]),hcos.ks,
                                   hcos.uk_profiles['nfw'][i,j], ellmax=self.lmax_out)
                kfft = kap*self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap,exp.pix).fft*self.ms_rescaled[j]

                if damp_1h_prof:
                    y_damp = tsz_filter * tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                       hcos.pk_profiles['y'][i, j]
                                                       *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                                       ellmax=exp.lmax)
                    u_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                     hcos.uk_profiles['nfw'][i, j]
                                     *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=exp.lmax)
                    u_sat_damp = self.CIB_satellite_filter[:, i, j] * u_damp

                    phicfft_ucen_usat_damp = exp.get_TT_qe(fftlog_way, ells_out, u_cen, u_sat_damp)
                    phicfft_usat_usat_damp = exp.get_TT_qe(fftlog_way, ells_out, u_sat_damp, u_sat_damp)
                    phicfft_ucen_y_damp = exp.get_TT_qe(fftlog_way, ells_out, u_cen, y_damp)
                    phicfft_usat_y_damp = exp.get_TT_qe(fftlog_way, ells_out, u_sat_damp, y_damp)
                    phicfft_yy_damp = exp.get_TT_qe(fftlog_way, ells_out, y_damp, y_damp)

                    kap_damp = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                            hcos.uk_profiles['nfw'][i, j]
                                            *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))), ellmax=self.lmax_out)
                    kfft_damp = kap_damp * self.ms_rescaled[j] if fftlog_way else ql.spec.cl2cfft(kap_damp, exp.pix).fft * self.ms_rescaled[j]
                else:
                    kfft_damp=kfft; phicfft_yy_damp=phicfft_yy; phicfft_usat_y_damp=phicfft_usat_y;
                    phicfft_ucen_y_damp=phicfft_ucen_y; phicfft_usat_usat_damp=phicfft_usat_usat;
                    phicfft_ucen_usat_damp=phicfft_ucen_usat; u_sat_damp=u_sat; y_damp=y


                # Accumulate the itgnds
                itgnd_1h_cross[...,j] = hcos.nzm[i,j] \
                                        * (phicfft_ucen_y_damp + phicfft_usat_y_damp) * np.conjugate(kfft_damp)
                itgnd_1h_Iyyy[...,j] = hcos.nzm[i,j] \
                                       * (phicfft_ucen_y_damp + phicfft_usat_y_damp) * np.conjugate(phicfft_yy_damp)
                itgnd_1h_IIyy[...,j] = hcos.nzm[i,j]\
                                       * np.conjugate(phicfft_yy_damp) * (phicfft_usat_usat_damp
                                                                          + 2 * phicfft_ucen_usat_damp)
                itgnd_1h_IyIy[...,j] = hcos.nzm[i,j] \
                                       * phicfft_usat_y_damp * (2*phicfft_ucen_y_damp + phicfft_usat_y_damp)
                itgnd_1h_yIII[...,j] = hcos.nzm[i,j] \
                                       * (phicfft_ucen_y_damp*phicfft_usat_usat_damp
                                          + phicfft_usat_y_damp*(2*phicfft_ucen_usat_damp + phicfft_usat_usat_damp) )

                itgnd_2h_k[...,j] = np.conjugate(kfft) * hcos.nzm[i,j] * hcos.bh_ofM[i,j]
                itgnd_2h_Iy[...,j] = (phicfft_ucen_y + phicfft_usat_y) * hcos.nzm[i,j] * hcos.bh_ofM[i,j]
                itgnd_2h_yy[...,j] = phicfft_yy * hcos.nzm[i,j] * hcos.bh_ofM[i,j]
                itgnd_2h_II[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] * (phicfft_usat_usat + 2 * phicfft_ucen_usat)

                itgnd_2h_Iyyy[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] * (phicfft_Iint_y*phicfft_yy
                                                                       + 3*phicfft_yint_y*(phicfft_ucen_y
                                                                                           +phicfft_usat_y))
                itgnd_2h_IIyy[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] * (2*phicfft_yy*(phicfft_Iint_ucen
                                                                                     +phicfft_Iint_usat)
                                                                       +2*phicfft_yint_y*(2*phicfft_ucen_usat
                                                                                          +phicfft_usat_usat))
                itgnd_2h_IyIy[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] * (2*phicfft_Iint_y*(phicfft_ucen_y
                                                                                         +phicfft_usat_y)
                                                                       +2*phicfft_usat_y*(2*phicfft_Iint_ucen
                                                                                          +phicfft_Iint_usat))
                itgnd_2h_yIII[...,j] = hcos.nzm[i,j] * hcos.bh_ofM[i,j] * (phicfft_yint_ucen*phicfft_usat_usat
                                                                       +(phicfft_yint_usat+phicfft_Iint_y)
                                                                       *(2*phicfft_ucen_usat +phicfft_usat_usat)
                                                                       +2*phicfft_Iint_ucen*phicfft_usat_y
                                                                       +2*phicfft_Iint_usat*(phicfft_ucen_y
                                                                                             +phicfft_usat_y))

                if get_secondary_bispec_bias:
                    # Temporary secondary bispectrum bias stuff
                    # The part with the nested lensing reconstructions
                    exp_param_dict = {'lmax': exp.lmax, 'nx': exp.nx, 'dx_arcmin': exp.dx*60.*180./np.pi}
                    # Get the kappa map, up to lmax rather than lmax_out as was needed in other terms
                    if damp_1h_prof:
                        kap_secbispec = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                     hcos.uk_profiles['nfw'][i, j]
                                                     *(1 - np.exp(-(hcos.ks / hcos.p['kstar_damping']))),
                                                     ellmax=exp.lmax)
                    else:
                        kap_secbispec = tls.pkToPell(hcos.comoving_radial_distance(hcos.zs[i]), hcos.ks,
                                                     hcos.uk_profiles['nfw'][i, j], ellmax=exp.lmax)
                    sec_bispec_rec = sbbs.get_sec_bispec_bias(lbins_sec_bispec_bias, qe_norm_1D,
                                                              exp_param_dict, exp.cltt_tot, u_cen,
                                                              kap_secbispec*self.ms_rescaled[j],
                                                              projected_fg_profile_2 = y_damp,
                                                              parallelise=parallelise_secondbispec)\
                                     + sbbs.get_sec_bispec_bias(lbins_sec_bispec_bias, qe_norm_1D, exp_param_dict,
                                                                exp.cltt_tot, u_sat_damp,
                                                                kap_secbispec*self.ms_rescaled[j],
                                                                projected_fg_profile_2 = y_damp,
                                                                parallelise=parallelise_secondbispec)
                    # TODO: I deleted the function definining hod_fact_1gal! Restore it
                    itgnd_1h_second_bispec[..., j] = hod_fact_1gal[i, j] * hcos.nzm[i,j] * sec_bispec_rec

            # Perform the m integrals
            Iyyy_1h[...,i]=np.trapz(itgnd_1h_Iyyy,hcos.ms,axis=-1)
            IIyy_1h[...,i]=np.trapz(itgnd_1h_IIyy,hcos.ms,axis=-1)
            IyIy_1h[...,i]=np.trapz(itgnd_1h_IyIy,hcos.ms,axis=-1)
            yIII_1h[...,i]=np.trapz(itgnd_1h_yIII,hcos.ms,axis=-1)

            oneH_cross[...,i]=np.trapz(itgnd_1h_cross,hcos.ms,axis=-1)

            if get_secondary_bispec_bias:
                oneH_second_bispec[...,i]=np.trapz(itgnd_1h_second_bispec,hcos.ms,axis=-1)

            # Accumulate integrands for 1-3 2-halo trispectrum
            Iyyy_2h_1_3[...,i] = np.trapz(itgnd_2h_Iyyy,hcos.ms,axis=-1)
            IIyy_2h_1_3[...,i] = np.trapz(itgnd_2h_IIyy,hcos.ms,axis=-1)
            IyIy_2h_1_3[...,i] = np.trapz(itgnd_2h_IyIy,hcos.ms,axis=-1)
            yIII_2h_1_3[...,i] = np.trapz(itgnd_2h_yIII,hcos.ms,axis=-1)

            # Accumulate integrands for 2-2 2-halo trispectrum
            Iyyy_2h_2_2[...,i] = 2 * np.trapz(itgnd_2h_Iy,hcos.ms,axis=-1)\
                                 * np.trapz(itgnd_2h_yy,hcos.ms,axis=-1) * pk_of_L
            IIyy_2h_2_2[...,i] = 2 * np.trapz(itgnd_2h_II,hcos.ms,axis=-1)\
                                 * np.trapz(itgnd_2h_yy,hcos.ms,axis=-1) * pk_of_L
            IyIy_2h_2_2[...,i] = 2 * np.trapz(itgnd_2h_Iy,hcos.ms,axis=-1)**2 * pk_of_L
            yIII_2h_2_2[...,i] = 2 *  np.trapz(itgnd_2h_Iy,hcos.ms,axis=-1)\
                                 * np.trapz(itgnd_2h_II,hcos.ms,axis=-1) * pk_of_L

            tmpCorr =np.trapz(itgnd_2h_k,hcos.ms,axis=-1)
            twoH_cross[...,i]=np.trapz(itgnd_2h_Iy,hcos.ms,axis=-1) * (tmpCorr + self.m_consistency[i]) * pk_of_L

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
            exp.biases['mixed']['second_bispec']['1h'] = np.trapz( oneH_second_bispec * kIy_itgnd, hcos.zs, axis=-1)
            exp.biases['second_bispec_bias_ells'] = lbins_sec_bispec_bias

        if fftlog_way:
            exp.biases['ells'] = np.arange(self.lmax_out+1)
            return
        else:
            exp.biases['ells'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['1h']).get_ml(lbins).ls
            exp.biases['mixed']['trispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['trispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['trispec']['2h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['prim_bispec']['1h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['prim_bispec']['1h']).get_ml(lbins).specs['cl']
            exp.biases['mixed']['prim_bispec']['2h'] = ql.maps.cfft(exp.nx,exp.dx,fft=exp.biases['mixed']['prim_bispec']['2h']).get_ml(lbins).specs['cl']
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
            self.get_cib_bias(exp, get_secondary_bispec_bias=get_secondary_bispec_bias)
            which_bias_list.append('cib')
        if get_tsz:
            self.get_tsz_bias(exp, get_secondary_bispec_bias=get_secondary_bispec_bias)
            which_bias_list.append('tsz')
        if get_mixed:
            self.get_cib_bias(exp, get_secondary_bispec_bias=get_secondary_bispec_bias)
            which_bias_list.append('mixed')

        for which_bias in which_bias_list:
            for which_coupling in which_coupling_list:
                if which_coupling=='second_bispec':
                    ells = exp.biases['second_bispec_bias_ells']
                    which_term_list = ['1h']
                    # TODO: Implement 2h term for sec bispec bias and incorporate into delensing bias calculation
                    print('2halo term not yet implemented for secondary bispectrum bias!')
                else:
                    ells = exp.biases['ells']
                    which_term_list = ['1h', '2h']

                for which_term in which_term_list:
                    clkk_bias_tot += np.nan_to_num(np.interp(np.arange(lmax_clkk+1), ells,
                                               exp.biases[which_bias][which_coupling][which_term]))
                    if which_coupling=='prim_bispec':
                        # The prim bispec bias to the recXtrueKappa is half of the prim bispec bias to the auto
                        clkcross_bias_tot += np.nan_to_num(0.5 * np.interp(np.arange(lmax_clkk+1), ells,
                                                             exp.biases[which_bias][which_coupling][which_term]))

        # Now use clkk_bias_tot to get the bias to C_l^{B^{template}x\tilde{B}} and C_l^{B^{template}xB^{template}}
        # TODO: speed up calculate_cl_bias() by using fftlog
        # TODO: It might be better to have W_E and W_phi provided externally rather than calculated internally
        cl_Btemp_x_Blens_bias_bcl = tls.calculate_cl_bias(exp.pix, exp.W_E * exp.cl_unl.clee, exp.W_phi * clkcross_bias_tot, lbins)
        cl_Btemp_x_Btemp_bias_bcl = tls.calculate_cl_bias(exp.pix, exp.W_E**2 * (exp.cl_len.clee + exp.nlpp),
                                                          exp.W_phi**2 * clkk_bias_tot, lbins)
        cl_Bdel_x_Bdel_bias_array = - 2 * cl_Btemp_x_Blens_bias_bcl.specs['cl'] + cl_Btemp_x_Btemp_bias_bcl.specs['cl']

        return cl_Btemp_x_Blens_bias_bcl.ls, cl_Btemp_x_Blens_bias_bcl.specs['cl'],\
               cl_Btemp_x_Btemp_bias_bcl.specs['cl'], cl_Bdel_x_Bdel_bias_array